import torch
import torch.nn as nn

import sklearn
import dgl
import dgllife
from dgllife.model import MPNNGNN
from model_utils import (
    pack_atom_feats,
    unpack_atom_feats,
    pack_bond_feats,
    unpack_bond_feats,
    reactive_pooling,
    Global_Reactivity_Attention,
    GELU,
)


class LocalTransform(nn.Module):
    def __init__(
        self,
        node_in_feats,
        edge_in_feats,
        node_out_feats,
        edge_hidden_feats,
        num_step_message_passing,
        attention_heads,
        attention_layers,
        Template_rn,
        Template_vn,
    ):
        super(LocalTransform, self).__init__()

        self.activation = GELU()

        self.mpnn = MPNNGNN(
            node_in_feats=node_in_feats,
            node_out_feats=node_out_feats,
            edge_in_feats=edge_in_feats,
            edge_hidden_feats=edge_hidden_feats,
            num_step_message_passing=num_step_message_passing,
        )

        self.atom_att = Global_Reactivity_Attention(
            node_out_feats, attention_heads, attention_layers, 8
        )
        self.bond_att = Global_Reactivity_Attention(
            node_out_feats, attention_heads, attention_layers, 2
        )

        self.pooling_v = nn.Sequential(
            nn.Linear(node_out_feats * 2, node_out_feats),
            self.activation,
            nn.Dropout(0.2),
            nn.Linear(node_out_feats, 2),
        )

        self.pooling_r = nn.Sequential(
            nn.Linear(node_out_feats * 2, node_out_feats),
            self.activation,
            nn.Dropout(0.2),
            nn.Linear(node_out_feats, 2),
        )

        self.bondnet_v = nn.Sequential(
            nn.Linear(node_out_feats * 2, node_out_feats),
            self.activation,
            nn.Linear(node_out_feats, node_out_feats),
        )

        self.bondnet_r = nn.Sequential(
            nn.Linear(node_out_feats * 2, node_out_feats),
            self.activation,
            nn.Linear(node_out_feats, node_out_feats),
        )

        self.output_v = nn.Sequential(
            nn.Linear(node_out_feats, node_out_feats),
            self.activation,
            nn.Dropout(0.2),
            nn.Linear(node_out_feats, Template_vn + 1),
        )

        self.output_r = nn.Sequential(
            nn.Linear(node_out_feats, node_out_feats),
            self.activation,
            nn.Dropout(0.2),
            nn.Linear(node_out_feats, Template_rn + 1),
        )

        self.poolings = {"virtual": self.pooling_v, "real": self.pooling_r}
        self.bondnets = {"virtual": self.bondnet_v, "real": self.bondnet_r}

    def forward(self, bg, adms, bonds_dict, node_feats, edge_feats):

        atom_feats = self.mpnn(bg, node_feats, edge_feats)
        atom_feats, mask = pack_atom_feats(bg, atom_feats)
        atom_feats, atom_attention = self.atom_att(atom_feats, adms, mask)
        idxs_dict, rout_dict, bonds_feats, bonds = reactive_pooling(
            bg, atom_feats, bonds_dict, self.poolings, self.bondnets
        )
        bond_feats, mask, bcms = pack_bond_feats(bonds_feats, bonds)
        bond_feats, bond_attention = self.bond_att(bond_feats, bcms, mask)
        feats_v, feats_r = unpack_bond_feats(bond_feats, idxs_dict)
        template_v, template_r = self.output_v(feats_v), self.output_r(feats_r)
        # template_v: topk x num virtual templates
        # template_r: topk x num real templates
        # rout_dict: learned topk bonds x 2
        # idxs_dict: list of topk indices
        return (
            template_v,
            template_r,
            rout_dict["virtual"],
            rout_dict["real"],
            idxs_dict["virtual"],
            idxs_dict["real"],
            (atom_attention, bond_attention),
        )


class ESMLocalTransform(LocalTransform):
    def __init__(
        self,
        node_in_feats,
        edge_in_feats,
        node_out_feats,
        edge_hidden_feats,
        num_step_message_passing,
        attention_heads,
        attention_layers,
        Template_rn,
        Template_vn,
    ):
        super(ESMLocalTransform, self).__init__(
            node_in_feats,
            edge_in_feats,
            node_out_feats,
            edge_hidden_feats,
            num_step_message_passing,
            attention_heads,
            attention_layers,
            Template_rn,
            Template_vn,
        )
        torch.hub.set_dir("/Mounts/rbg-storage1/snapshots/metabolomics/esm2")
        self.esm_model, self.alphabet = torch.hub.load(
            "facebookresearch/esm", "esm2_t12_35M_UR50D"
        )
        self.esm_tokenizer = self.alphabet.get_batch_converter()

        self.mha = nn.MultiheadAttention(
            embed_dim=self.esm_model.embed_dim,
            num_heads=attention_heads,
            batch_first=True,
        )
        self.lin = nn.Linear(2 * node_out_feats, node_out_feats)

    def forward(self, bg, adms, bonds_dict, node_feats, edge_feats, sequences):

        atom_feats = self.mpnn(bg, node_feats, edge_feats)

        # encode protein
        protein_feats, protein_mask = self.encode_sequence(sequences)
        dense_cs, cs_mask = pack_atom_feats(bg, atom_feats)  # TODO

        c_protein, _ = (
            self.mha(  # batch_size, max_num_nodes (reactants), hidden dim # TODO
                query=dense_cs,
                key=protein_feats,
                value=protein_feats,
                key_padding_mask=protein_mask.float(),
            )
        )
        c_protein = c_protein[cs_mask.bool()]  # TODO
        atom_feats = self.lin(torch.cat([atom_feats, c_protein], dim=-1))
        atom_feats, mask = pack_atom_feats(bg, atom_feats)  # TODO
        atom_feats, atom_attention = self.atom_att(atom_feats, adms, mask)
        idxs_dict, rout_dict, bonds_feats, bonds = reactive_pooling(
            bg, atom_feats, bonds_dict, self.poolings, self.bondnets
        )
        bond_feats, mask, bcms = pack_bond_feats(bonds_feats, bonds)
        bond_feats, bond_attention = self.bond_att(bond_feats, bcms, mask)
        feats_v, feats_r = unpack_bond_feats(bond_feats, idxs_dict)
        template_v, template_r = self.output_v(feats_v), self.output_r(feats_r)
        # template_v: topk x num virtual templates
        # template_r: topk x num real templates
        # rout_dict: learned topk bonds x 2
        # idxs_dict: list of topk indices
        return (
            template_v,
            template_r,
            rout_dict["virtual"],
            rout_dict["real"],
            idxs_dict["virtual"],
            idxs_dict["real"],
            (atom_attention, bond_attention),
        )

    def encode_sequence(self, sequences):
        fair_x = [(i, s) for i, s in enumerate(sequences)]
        _, _, batch_tokens = self.esm_tokenizer(fair_x)
        batch_tokens = batch_tokens.to(self.lin.weight.device)

        mask = batch_tokens != self.esm_model.cls_idx
        mask *= batch_tokens != self.esm_model.padding_idx
        mask *= batch_tokens != self.esm_model.eos_idx

        result = self.esm_model(
            batch_tokens,
            repr_layers=[self.esm_model.num_layers],
            return_contacts=False,
        )
        encoder_hidden_states = result["representations"][self.esm_model.num_layers]

        return encoder_hidden_states, mask
