import os

from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment


class FLUXEnvironment(ClusterEnvironment):
    @property
    def creates_processes_externally(self) -> bool:
        return True

    @property
    def main_address(self) -> str:
        root = os.environ.get("MASTER_ADDR")
        if root is None:
            root = "127.0.0.1"
            os.environ["MASTER_ADDR"] = root
        return root

    @property
    def main_port(self) -> int:
        if "MASTER_PORT" in os.environ:
            return int(os.environ["MASTER_PORT"])

        job_id = os.environ.get("NUMERIC_JOB_ID")
        port = int(job_id[-4:]) + 15000 if job_id else 12910
        os.environ["MASTER_PORT"] = str(port)
        return port

    def world_size(self) -> int:
        return int(os.environ["FLUX_JOB_SIZE"])

    def set_world_size(self, size: int) -> None:
        return None

    def global_rank(self) -> int:
        return int(os.environ["FLUX_TASK_RANK"])

    def set_global_rank(self, rank: int) -> None:
        return None

    def local_rank(self) -> int:
        return int(os.environ["FLUX_TASK_LOCAL_ID"])

    def node_rank(self) -> int:
        world = int(os.environ["FLUX_JOB_SIZE"])
        nodes = int(os.environ["FLUX_JOB_NNODES"])
        return int(os.environ["FLUX_TASK_RANK"]) // (world // nodes)

    def validate_settings(self, num_devices: int, num_nodes: int) -> None:
        tasks_per_node = int(os.environ["FLUX_JOB_SIZE"]) // int(os.environ["FLUX_JOB_NNODES"])
        if tasks_per_node != num_devices:
            raise ValueError(
                f"devices={num_devices} but Flux tasks-per-node={tasks_per_node}. "
                f"Set trainer.devices={tasks_per_node}."
            )

        actual_nodes = int(os.environ["FLUX_JOB_NNODES"])
        if actual_nodes != num_nodes:
            raise ValueError(
                f"num_nodes={num_nodes} but Flux nodes={actual_nodes}. "
                f"Set trainer.num_nodes={actual_nodes}."
            )
