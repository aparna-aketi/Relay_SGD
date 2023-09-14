
"""
Gossipers

:description: Gossiper's are designed for multi-peer communication (i.e., send
              and recv from multiple peers at each ieration)
"""

import torch
import torch.distributed as dist
import copy
from .graph_manager import GraphManager


class dist_backend:
    UNDEFINED = -1
    TCP = 0
    MPI = 1
    GLOO = 2
    NCCL = 3


class Gossiper(object):
    """ Generic gossip averaging object for multi-peer communication """

    def __init__(self, msg, graph, device=None, logger=None,
                 rank=None, world_size=None):
        """
        Initialize generic averaging class designed for multi-peer comms

        :param msg: (tensor) message used to initialize recv buffer
        :param device: (device) device on which to initialize recv buffer
        :param graph: (GraphManager) Subclass of GraphManager
        :param logger: (python logger) module used to log results
        """

        self.logger = logger
        if rank is None or world_size is None:
            assert dist.is_initialized()
            # for now p2p communication only supported withed tcp and mpi
            assert dist._backend != dist_backend.GLOO
            assert dist._backend != dist_backend.NCCL
            rank = dist.get_rank()
            world_size = dist.get_world_size()

        # graph topology properties
        self.rank       = rank
        self.world_size = world_size
        assert isinstance(graph, GraphManager)
        self._graph_manager       = graph
        self.peers_per_itr_device = torch.tensor(
            [self._graph_manager.peers_per_itr], device=device,
            dtype=msg.dtype)
        self.passive = self._graph_manager.is_passive()
        self.refresh_peers_()  # sets in- and out-peers attributes

        # msg buffers used during send/recv
        self.device = device if device is not None else msg.device
        self.out_msg_buffer = []
        self.mji            = {}
        self.cji            = {}
        self.in_msg_buffer = msg.clone().detach_().to(self.device)
        for in_edge in self.in_edges:
            self.mji[in_edge.src] = torch.zeros(len(self.in_msg_buffer)-1, device=device)
            self.cji[in_edge.src] = torch.zeros(1, device=device)
            
        if self.device.type == 'cpu':
            try:
                self.in_msg_buffer = self.in_msg_buffer.pin_memory()
            except Exception as e:
                if self.logger is not None:
                    self.logger.error(e)
        
        self._pending_req = None

    @property
    def peers_per_itr(self):
        return self._graph_manager.peers_per_itr

    @peers_per_itr.setter
    def peers_per_itr(self, v):
        self._graph_manager.peers_per_itr = v

    def refresh_peers_(self):
        """ Update in- and out-peers """
        
        self.out_edges, self.in_edges = self._graph_manager.get_edges()

    def clean_msg_buffers_(self):
        """ Clean outgoing message buffer """
        msgs = []
        while len(self.out_msg_buffer) > 0:
            req, msg = self.out_msg_buffer.pop()
            req.wait()
            msgs.append(msg)
        while len(msgs) > 0:
            msg = msgs.pop()
            with torch.no_grad():
                msg.set_()

    def parse_in_msg_buffer(self, residual=False):
        """ Parse in-msg buffer and return msg and ps-weight separately """
        msg = self.in_msg_buffer
        if not self.regular:
            return msg.narrow(0, 0, len(msg) - 1), msg[-1]
        else:
            return msg
    
    def mix_out_msg_(self, out_msg, out_edge):
        msg = copy.deepcopy(out_msg)
        c   = torch.ones(1, device=self.device)
        for key, mji in self.mji.items():
            if key != out_edge:
                msg += copy.deepcopy(mji)   # m_ij = x_i + sum(m_ki)
                c   += copy.deepcopy(self.cji[key])
        
        msg = torch.cat([msg, c.type(out_msg.dtype)])

        return msg
            
    def mix(self):
        """ Single gossip step """
        raise NotImplementedError

class Relay(Gossiper):

    def mix(self, out_msg):
        """ Consensus averaging step """
        # out_msg must be on the correct device
        assert out_msg.device.type == self.device.type
        if self.logger is not None:
            self.logger.debug('in/out -peers {}/{}'
                              .format(self.in_edges, self.out_edges))

        # prepare messages for gossip
        placeholder = torch.zeros_like(torch.cat([out_msg, torch.zeros(1, device=self.device).type(out_msg.dtype)]))
        
        # non-blocking send
        # print(len(self.out_edges), len(self.in_edges))
        data_amt = 0
        for out_edge in self.out_edges:
            assert self.rank == out_edge.src 
            #print("rank, out", self.rank, out_edge.dest)
            msg = self.mix_out_msg_(out_msg, out_edge.dest)
            #print(self.rank, 'sends', out_edge.dest, 'value of', msg[-1])
            req = dist.broadcast(tensor=msg, src=out_edge.src, group=out_edge.process_group, async_op=True)
            self.out_msg_buffer.append((req, msg))
            data_amt += msg.element_size()*msg.nelement()
        # blocking recv w/ some code optimization to avoid buffer prep overhead

        self.in_msg_buffer.zero_()
        for in_edge in self.in_edges:
            #print("rank, in", self.rank, in_edge.src)
            dist.broadcast(tensor=placeholder, src=in_edge.src, group=in_edge.process_group)
            self.mji[in_edge.src] = copy.deepcopy(placeholder).narrow(0, 0, len(placeholder) - 1)
            self.cji[in_edge.src] = copy.deepcopy(placeholder[-1])
            #print(self.rank, 'recieves from', in_edge.src, 'value of', placeholder[-1])
            self.in_msg_buffer.add_(placeholder)
                
        self.refresh_peers_()
        self.clean_msg_buffers_()
        in_msg = self.in_msg_buffer.narrow(0, 0, len(self.in_msg_buffer)-1)
        n = 1.0 + self.in_msg_buffer[-1]

        return in_msg, n, data_amt
    
