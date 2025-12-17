# !/usr/bin/env python
"""
Created by Indraneel on 12/15/2025

Actor process to interact with environment with current policy and collect interactions

"""
import os
import logging
from functools import lru_cache
import grpc
import time

from torch import nn
from torch.multiprocessing import Event, Queue

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.utils.utils import (
    init_logging
)
from lerobot.rl.process import ProcessSignalHandler
from lerobot.transport.utils import (
    grpc_channel_options,
    receive_bytes_in_chunks
)
from lerobot.transport import services_pb2, services_pb2_grpc

def use_threads(cfg: TrainRLServerPipelineConfig) -> bool:
    return cfg.policy.concurrency.actor == "threads"

@parser.wrap()
def actor_cli(cfg: TrainRLServerPipelineConfig):
    cfg.validate()
    display_pid = False
    if not use_threads(cfg):
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")
        display_pid = True
    
    # Create logs directory
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"actor_{cfg.job_name}.log")

    #Initialize logging with explicit log file
    init_logging(log_file=log_file, display_pid=display_pid)
    logging.info(f"Actor logging initialized, writing to {log_file}")

    is_threaded = use_threads(cfg)
    shutdown_event = ProcessSignalHandler(is_threaded, display_pid=display_pid).shutdown_event

    # Create the grpc server 
    learner_client, grpc_channel = learner_service_client(
        host=cfg.policy.actor_learner_config.learner_host.
        port=cfg.policy.actor_learner_config.learner_port,
    )

    logging.info("[ACTOR] Establishing connection with Learner")
    if not establish_learner_connection(learner_client, shutdown_event):
        logging.error("[ACTOR] Failed to establish connection with Learner")
        return

    if not use_threads(cfg):
        # If we use multithreading we can reuse the channel
        grpc_channel.close()
        grpc_channel = None
    
    logging.info("[ACTOR] Connection with Learner established")

    parameters_queue = Queue()
    transitions_queue = Queue()
    interactions_queue = Queue()

    # Set up either threads or processes
    concurrency_entity = None
    if use_threads(cfg):
        from threading import Thread

        concurrency_entity = Thread
    else:
        from multiprocessing import Process

        concurrency_entity = Process

    receive_policy_process = concurrency_entity(
        target=receive_policy,
        args=(cfg, parameters_queue, shutdown_event, grpc_channel),
        daemon=True
    )

    transitions_process = concurrency_entity(
        target=send_transitions,
        args=(cfg, transitions_queue, shutdown_event, grpc_channel),
        daemon=True
    )

    interactions_process = concurrency_entity(
        target=send_interactions,
        args=(cfg, interactions_queue, shutdown_event, grpc_channel),
        daemon=True,
    )

    transitions_process.start()
    interactions_process.start()
    receive_policy_process.start()

    act_with_policy(
        cfg=cfg,
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        transitions_queue=transitions_queue,
        interactions_queue=interactions_queue
    )
    logging.info("[ACTOR] Policy process joined")

    logging.info("[ACTOR] Closing queues")
    transitions_queue.close()
    interactions_queue.close()
    parameters_queue.close()

    transitions_process.join()
    logging.info("[ACTOR] Transitions process joined")
    interactions_process.join()
    logging.info("[ACTOR] Interactions process joined")
    receive_policy_process.join()
    logging.info("[ACTOR] Receive policy process joined")

    logging.info("[ACTOR] Join queues")
    transitions_queue.cancel_join_thread()
    interactions_queue.cancel_join_thread()
    parameters_queue.cancel_join_thread()

    logging.info("[ACTOR] queues closed")
    



def establish_learner_connection(
    stub: services_pb2_grpc.LearnerServiceStub,
    shutdown_event: Event, # type: ignore
    attempts: int = 30,
):
    """
        Establish a connection with learner
    """
    for _ in range(attempts):
        if shutdown_event.is_set():
            logging.info("[ACTOR] Shutting down establish_learner_connection")
            return False

        # Force a connection attempt and check state
        try:
            logging.info("[ACTOR] Send ready message to Learner")
            if stub.Ready(services_pb2.Empty()) == services_pb2.Empty():
                return True
        except grpc.RpcError as e:
            logging.error(f"[ACTOR] Waiting for Learner to be ready... {e}")
            time.sleep(2)
    return False 

# Run function only once and cache the result
@lru_cache(max_size=1)
def learner_service_client(
    host: str = "127.0.0.1",
    port: int = 50051,
) -> tuple[services_pb2_grpc.LearnerServiceStub, grpc.Channel]:
    """
    Returns a client for the learner service
    """
    channel = grpc.insecure_channel(
       f"{host}:{port}",
       grpc_channel_options() 
    )
    stub = services_pb2_grpc.LearnerServiceStub(channel)
    logging.info("[ACTOR] Learner service client created")
    return stub, channel


def receive_policy(
        cfg: TrainRLServerPipelineConfig,
        parameters_queue: Queue,
        shutdown_event: Event, # type: ignore
        learner_client: services_pb2_grpc.LearnerServiceStub | None = None,
        grpc_channel: grpc.Channel | None = None,
):
    """ Receive parameters from the learner"""
    logging.info("[ACTOR] Start receiving parameters from the learner")
    if not use_threads(cfg):
        # Create a process specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"actor_receive_policy_{os.get_pid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Actor receive policy process logging initialized")

        # Setup process handlers
        _ = ProcessSignalHandler(use_threads=False, display_pid=True)

    if grpc_channel is None or learner_client is None:
        learner_client, grpc_channel = learner_service_client(
            host=cfg.policy.actor_learner_config.learner_host,
            port=cfg.policy.actor_learner_config.learner_port,
        )
    
    try:
        iterator = learner_client.StreamParameters(services_pb2.Empty())
        receive_bytes_in_chunks(
            iterator,
            parameters_queue,
            shutdown_event,
            log_prefix="[ACTOR] parameters"
        )
    except grpc.RpcError as e:
        logging.error(f"[ACTOR] gRPC error: {e}")

    if not use_threads(cfg):
        grpc_channel.close()
    logging.info("[ACTOR] Received policy loop stopped")



if __name__ == "__main__":
    actor_cli()