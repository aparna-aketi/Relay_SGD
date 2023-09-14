
"""
Graph Manager Class

:description: Class provides an API for loading different peer-to-peer
    communication topologies, and cycling through peers.
"""

from math import log as mlog
import torch
import torch.distributed as dist

from .utils import is_power_of


class Edge(object):

    def __init__(self, local_master_rank, dest, src, local_rank, devices):
        self.dest = dest
        self.src = src
        self.process_group = dist.new_group([src, dest])
        if local_master_rank in [self.src, self.dest] and local_rank == 0:
            initializer_tensor = torch.Tensor([1]).to(torch.device("cuda:{}".format(local_master_rank%devices)))
            dist.all_reduce(initializer_tensor, group=self.process_group)
            initializer_tensor = torch.Tensor([1]).to(torch.device("cuda:{}".format(local_master_rank%devices))).half()
            dist.all_reduce(initializer_tensor, group=self.process_group)


class GraphManager(object):

    def __init__(self, rank, world_size, devices, local_rank=0):
        self.rank = rank
        self.world_size = world_size
        self.phone_book = [[] for _ in range(self.world_size)]
        self.local_rank = local_rank
        self.devices = devices
        self._make_graph()

    @property
    def peers_per_itr(self):
        return self._peers_per_itr

    @peers_per_itr.setter
    def peers_per_itr(self, v):
        self._peers_per_itr = v
        # set group-indices attr. --- point to out-peers in phone-book
        self._group_indices = [i for i in range(v)]

    def _make_graph(self):
        """
        Returns a nested list of peers; the outer-list is indexed by rank,
        the inner list denotes the set of peers that 'rank' can send
        messages to at any point in time
        """
        raise NotImplementedError

    def _add_peers(self, rank, peers):
        if rank == self.rank:
            self.peers_per_itr  = len(peers)
            self._group_indices = [i for i in range(self.peers_per_itr)]
        #print(rank, peers)
        for peer in peers:
            if peer not in self.phone_book[rank]:
                self.phone_book[rank].append(Edge(
                    local_master_rank=(self.rank),
                    dest=(peer),
                    src=(rank),
                    local_rank=self.local_rank, devices=self.devices))
        #print(rank, self.peers_per_itr)
            

    def is_regular_graph(self):
        """ Whether each node has the same number of in-peers as out-peers """
        raise NotImplementedError

    def is_bipartite_graph(self):
        """ Whether graph is bipartite or not """
        raise NotImplementedError

    def is_passive(self, rank=None):
        """ Whether 'rank' is a passive node or not """
        raise NotImplementedError

    def is_dynamic_graph(self, graph_type=None):
        """ Whether the graph-type is dynamic (as opposed to static) """
        raise NotImplementedError

    def get_peers(self):
        """ Returns the out and in-peers corresponding to 'self.rank' """
        # get out- and in-peers using new group-indices
        out_peers, in_peers = [], []
        for group_index in self._group_indices:
            out_peers.append(self.phone_book[self.rank][group_index].dest)
            in_peers.append(self.phone_book[self.rank][group_index].dest)
        return out_peers, in_peers

    def get_edges(self):
        """ Returns the pairwise process groups between rank and the out and
        in-peers corresponding to 'self.rank' """

        # get out- and in-peers using new group-indices
        out_edges, in_edges = [], []
        for group_index in self._group_indices:
            out_edges.append(self.phone_book[self.rank][group_index])
            #in_edges.append(self.phone_book[self.rank][group_index])
        
        for group_index in range(2):
            for rank, edges in enumerate(self.phone_book):
                if rank == self.rank:
                    continue
                try:
                    if self.rank == edges[group_index].dest:
                        in_edges.append(self.phone_book[rank][group_index])
                except:
                    continue
        return out_edges, in_edges


    def _rotate_forward(self, r, p):
        """ Helper function returns peer that is p hops ahead of r """
        return (r + p) % self.world_size

    def _rotate_backward(self, r, p):
        """ Helper function returns peer that is p hops behind r """
        temp = r
        for _ in range(p):
            temp -= 1
            if temp < 0:
                temp = self.world_size - 1
        return temp



class ChainGraph(GraphManager):

    def _make_graph(self):
        for rank in range(self.world_size):
            if rank == 0:
                f_peer = self._rotate_forward(rank, 1)
                self._add_peers(rank, [f_peer])
            elif rank == self.world_size -1:
                b_peer = self._rotate_backward(rank, 1)
                self._add_peers(rank, [b_peer])
            else:
                f_peer = self._rotate_forward(rank, 1)
                b_peer = self._rotate_backward(rank, 1)
                self._add_peers(rank, [f_peer, b_peer])
        #print("edges", self.rank, self.get_peers(), self.peers_per_itr, self._group_indices)

    def is_regular_graph(self): return True

    def is_bipartite_graph(self): return False

    def is_passive(self, rank=None): return False

    def is_dynamic_graph(self, graph_type=None): return False