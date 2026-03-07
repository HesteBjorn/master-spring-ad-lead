"""
Script that starts n CARLA servers, n leaderboard clients and distributed TFv6 PPO training.
Adapted with minimal changes from CaRL's team_code/train_parallel.py.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import shlex
import socket
import subprocess
import sys
import time

import psutil


def strtobool(v):
    return str(v).lower() in ("yes", "y", "true", "t", "1", "True")


def next_free_port(port=1024, max_port=65535):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while port <= max_port:
        try:
            sock.bind(("", port))
            sock.close()
            return port
        except OSError:
            port += 1
    raise OSError("no free ports")


def kill(proc_pid):
    if psutil.pid_exists(proc_pid):
        process = psutil.Process(proc_pid)
        for proc in process.children(recursive=True):
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                pass
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass


def kill_all_carla_servers(ports):
    # Failsafe cleanup by ports, as in CaRL.
    for proc in psutil.process_iter():
        try:
            proc_connections = proc.connections(kind="all")
        except (PermissionError, psutil.AccessDenied, psutil.NoSuchProcess):
            proc_connections = None

        if proc_connections is not None:
            for conns in proc_connections:
                if not isinstance(conns.laddr, str):
                    if conns.laddr.port in ports:
                        try:
                            proc.kill()
                        except psutil.NoSuchProcess:
                            pass


def cleanup(carla_procs, leaderboard_procs, train_proc, carla_ports):
    kill_all_carla_servers(carla_ports)

    for carla_proc in carla_procs:
        kill(carla_proc.pid)

    for leaderboard_proc in leaderboard_procs:
        kill(leaderboard_proc.pid)

    if train_proc is not None:
        kill(train_proc.pid)


def town_to_route_prefix(town_id: int) -> str:
    mapping = {
        1: "Town01",
        2: "Town02",
        3: "Town03",
        4: "Town04",
        5: "Town05",
        6: "Town06",
        7: "Town07",
        10: "Town10HD",
        11: "Town11",
        12: "Town12",
        13: "Town13",
        15: "Town15",
    }
    if town_id not in mapping:
        raise ValueError(f"Unsupported town id: {town_id}")
    return mapping[town_id]


def build_route_files(args) -> list[str]:
    if args.route_files:
        if len(args.route_files) < args.num_envs_per_node:
            raise ValueError(
                f"--route_files has {len(args.route_files)} entries but "
                f"--num_envs_per_node is {args.num_envs_per_node}"
            )
        return [str(pathlib.Path(x).resolve()) for x in args.route_files]

    route_root_folder = os.path.join(
        args.git_root, f"custom_leaderboard/leaderboard/data/{args.routes_folder}"
    )
    route_start_id = args.num_envs_per_gpu * args.node_id
    route_end_id = args.route_end_id

    id_to_townfile_mapping = {}
    for town_id in set(args.train_towns):
        town_prefix = town_to_route_prefix(town_id)
        id_to_townfile_mapping[town_id] = [
            os.path.join(route_root_folder, f"route_{town_prefix}_{i:02d}.xml.gz")
            for i in range(route_start_id, route_end_id)
        ]

    route_files = []
    for town_id in args.train_towns:
        candidates = id_to_townfile_mapping[town_id]
        while candidates and not os.path.exists(candidates[0]):
            candidates.pop(0)
        if not candidates:
            raise FileNotFoundError(
                f"No route files found for town {town_id} in {route_root_folder} "
                f"(start={route_start_id}, end={route_end_id})"
            )
        route_files.append(candidates.pop(0))

    if len(route_files) < args.num_envs_per_node:
        raise ValueError(
            f"Need at least {args.num_envs_per_node} routes but got {len(route_files)}. "
            "Set --train_towns / --route_files accordingly."
        )
    return route_files


if __name__ == "__main__":
    try:
        training = True
        repo_root_default = str(pathlib.Path(__file__).resolve().parents[1])
        carl_root_default = str(
            pathlib.Path(repo_root_default) / "3rd_party" / "CaRL" / "CARLA"
        )
        tfv6_checkpoint_default = str(
            pathlib.Path(repo_root_default)
            / "outputs"
            / "checkpoints"
            / "tfv6_resnet34"
        )
        log_root_default = str(pathlib.Path(repo_root_default) / "outputs" / "rl_logs")

        parser = argparse.ArgumentParser(allow_abbrev=False)
        parser.add_argument(
            "--exp_name",
            type=str,
            default="TFV6_PPO_000",
            help="name of this experiment",
        )
        parser.add_argument(
            "--repo_root",
            type=str,
            default=repo_root_default,
            help="root folder of this repository",
        )
        parser.add_argument(
            "--git_root",
            type=str,
            default=carl_root_default,
            help="root folder of 3rd_party/CaRL/CARLA",
        )
        parser.add_argument(
            "--tfv6_checkpoint",
            type=str,
            default=tfv6_checkpoint_default,
            help="TFv6 checkpoint folder containing config.json and model*.pth",
        )
        parser.add_argument(
            "--log_root",
            type=str,
            default=log_root_default,
            help="root output folder for TFv6 PPO runs",
        )
        parser.add_argument(
            "--gpu_ids",
            nargs="+",
            default=[0],
            type=int,
            help="GPU ids for training ranks. Length must match world size.",
        )
        parser.add_argument(
            "--carla_root",
            default=os.environ.get("CARLA_ROOT", ""),
            type=str,
            help="Path to CARLA root folder containing CarlaUE4.sh",
        )
        parser.add_argument(
            "--start_port",
            default=1024,
            type=int,
            help="Lowest port to use. Increase if multiple launchers run on one machine.",
        )
        parser.add_argument(
            "--train_towns",
            nargs="+",
            default=(3,),
            type=int,
            help="Town ids used per environment.",
        )
        parser.add_argument(
            "--routes_folder",
            default="debug_routes_with_scenarios",
            type=str,
            help="Folder in custom_leaderboard/leaderboard/data/ containing route files.",
        )
        parser.add_argument(
            "--route_end_id",
            default=32,
            type=int,
            help="Upper exclusive route index to probe (kept for CaRL compatibility).",
        )
        parser.add_argument(
            "--route_files",
            nargs="+",
            default=None,
            help="Optional explicit route files (overrides --routes_folder and --train_towns).",
        )
        parser.add_argument(
            "--num_envs_per_gpu",
            default=1,
            type=int,
            help="Number of environments per GPU/process group shard.",
        )
        parser.add_argument(
            "--seed", type=int, default=0, help="seed of the experiment"
        )
        parser.add_argument(
            "--num_envs_per_node",
            default=1,
            type=int,
            help="Total number of environments to run on this node.",
        )
        parser.add_argument(
            "--num_nodes", default=1, type=int, help="Number of machines to train on."
        )
        parser.add_argument(
            "--node_id", default=0, type=int, help="Node id of this launcher."
        )
        parser.add_argument(
            "--rdzv_addr",
            default="localhost",
            type=str,
            help="Rendezvous address for torchrun.",
        )
        parser.add_argument(
            "--rdzv_port", default=0, type=int, help="Rendezvous port for torchrun."
        )
        parser.add_argument(
            "--ml_cloud",
            default=0,
            type=int,
            help="Whether to use slower process staggering for large clusters.",
        )
        parser.add_argument(
            "--use_traj_sync_ppo",
            type=lambda x: bool(strtobool(x)),
            default=False,
            nargs="?",
            const=True,
            help="If true run each env in separate process.",
        )
        parser.add_argument(
            "--route_repetitions",
            type=int,
            default=100,
            help="How often to repeat training routes.",
        )
        parser.add_argument(
            "--debug",
            type=lambda x: bool(strtobool(x)),
            default=False,
            nargs="?",
            const=True,
            help="if true, no auto-restart after crash and logs are written per process",
        )
        parser.add_argument(
            "--carla_singularity",
            type=lambda x: bool(strtobool(x)),
            default=False,
            nargs="?",
            const=True,
            help="whether to run CARLA via singularity",
        )
        parser.add_argument(
            "--carla_singularity_path",
            type=str,
            default="",
            help="path to CARLA singularity image",
        )
        parser.add_argument(
            "--track",
            type=str,
            default="MAP_QUALIFIER",
            help="Leaderboard track (e.g. MAP, MAP_QUALIFIER).",
        )
        parser.add_argument("--frame_rate", type=float, default=10.0)
        parser.add_argument("--timeout", type=float, default=900.0)
        parser.add_argument("--runtime_timeout", type=float, default=900.0)
        parser.add_argument(
            "--carla_rpc_threads",
            type=int,
            default=8,
            help="CARLA RPC worker threads per server.",
        )
        parser.add_argument(
            "--carla_streaming_threads",
            type=int,
            default=8,
            help="CARLA streaming worker threads per server.",
        )
        parser.add_argument(
            "--carla_secondary_threads",
            type=int,
            default=8,
            help="CARLA secondary worker threads per server.",
        )
        parser.add_argument("--leaderboard_debug", type=int, default=0)

        args, unknown = parser.parse_known_args()

        if not args.carla_root:
            raise ValueError(
                "--carla_root is required (or export CARLA_ROOT) so the launcher can start CARLA servers."
            )

        if args.num_envs_per_node <= 0:
            raise ValueError("--num_envs_per_node must be > 0")
        if args.num_envs_per_gpu <= 0:
            raise ValueError("--num_envs_per_gpu must be > 0")
        if (
            args.num_envs_per_node % args.num_envs_per_gpu != 0
            and not args.use_traj_sync_ppo
        ):
            raise ValueError(
                "--num_envs_per_node must be divisible by --num_envs_per_gpu when use_traj_sync_ppo=False"
            )
        if not args.gpu_ids:
            raise ValueError("--gpu_ids must not be empty")

        args.repo_root = str(pathlib.Path(args.repo_root).resolve())
        args.git_root = str(pathlib.Path(args.git_root).resolve())
        args.log_root = str(pathlib.Path(args.log_root).resolve())
        args.tfv6_checkpoint = str(pathlib.Path(args.tfv6_checkpoint).resolve())
        args.carla_root = str(pathlib.Path(args.carla_root).resolve())

        os.environ["WORK_DIR"] = args.git_root
        scenario_runner_root = os.path.join(
            args.git_root, "custom_leaderboard", "scenario_runner"
        )
        leaderboard_root = os.path.join(
            args.git_root, "custom_leaderboard", "leaderboard"
        )
        os.environ["SCENARIO_RUNNER_ROOT"] = scenario_runner_root
        os.environ["LEADERBOARD_ROOT"] = leaderboard_root

        pythonpath_entries = [
            os.path.join(args.carla_root, "PythonAPI"),
            os.path.join(args.carla_root, "PythonAPI", "carla"),
            scenario_runner_root,
            leaderboard_root,
            args.repo_root,
        ]
        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = ":".join(
            [x for x in pythonpath_entries + [existing_pythonpath] if x]
        )

        raw_logdir = args.log_root
        logdir = os.path.join(raw_logdir, args.exp_name)
        os.makedirs(logdir, exist_ok=True)
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "").strip()
        logs_folder_name = f"logs_{slurm_job_id}" if slurm_job_id else "logs"
        process_logdir = os.path.join(raw_logdir, logs_folder_name)
        os.makedirs(process_logdir, exist_ok=True)
        print(f"[launcher] process_logs={process_logdir}")

        route_files = build_route_files(args)

        blackhole = open(os.devnull, "w", encoding="utf-8")
        server_outs = []
        server_errs = []
        client_outs = []
        client_errs = []
        for i in range(args.num_envs_per_node):
            if args.debug:
                server_outs.append(
                    open(
                        os.path.join(process_logdir, f"server_out_{i:03d}.txt"),
                        "w",
                        encoding="utf-8",
                    )
                )
                server_errs.append(
                    open(
                        os.path.join(process_logdir, f"server_err_{i:03d}.txt"),
                        "w",
                        encoding="utf-8",
                    )
                )
                client_outs.append(
                    open(
                        os.path.join(process_logdir, f"client_out_{i:03d}.txt"),
                        "w",
                        encoding="utf-8",
                    )
                )
                client_errs.append(
                    open(
                        os.path.join(process_logdir, f"client_err_{i:03d}.txt"),
                        "w",
                        encoding="utf-8",
                    )
                )
            else:
                server_outs.append(blackhole)
                server_errs.append(blackhole)
                client_outs.append(blackhole)
                client_errs.append(blackhole)

        current_port = args.start_port + 5000 * args.node_id
        skip_next_route = "False"

        while training:
            if args.debug:
                training = False
            train_process = None
            rl_ports = []
            traffic_manager_ports = []
            sensor_ports = []
            client_ports = []
            carla_primary_ports = []
            if current_port > 60000:
                current_port = args.start_port
            for _ in range(args.num_envs_per_node):
                current_port = next_free_port(current_port)
                rl_ports.append(current_port)
                current_port += 3
                current_port = next_free_port(current_port)
                traffic_manager_ports.append(current_port)
                current_port += 3
                current_port = next_free_port(current_port)
                sensor_ports.append(current_port)
                current_port += 3
                current_port = next_free_port(current_port)
                client_ports.append(current_port)
                current_port += 3
                current_port = next_free_port(current_port)
                carla_primary_ports.append(current_port)
                current_port += 3

                if args.num_nodes > 1:
                    tcp_store_port = 7000
                else:
                    tcp_store_port = next_free_port(7000)

            carla_processes = []
            leaderboard_processes = []

            server_sleep = 7 if args.ml_cloud else 0.02
            client_sleep = 0.2 if args.ml_cloud else 0.02
            # TFv6 is a sensor agent. Keep CARLA threading enabled.
            tfv6_rpc_threads = args.carla_rpc_threads
            tfv6_streaming_threads = args.carla_streaming_threads
            tfv6_secondary_threads = args.carla_secondary_threads

            for i in range(args.num_envs_per_node):
                print(f"Start server {i}")
                graphics_adapter = args.gpu_ids[i % len(args.gpu_ids)]
                if args.carla_singularity:
                    if not args.carla_singularity_path:
                        raise ValueError(
                            "--carla_singularity_path is required when --carla_singularity is set."
                        )
                    carla_processes.append(
                        subprocess.Popen(
                            f"singularity exec --nv --bind {args.carla_root}:{args.carla_root},{raw_logdir}:{raw_logdir} "
                            f"{args.carla_singularity_path} "
                            f"bash {args.carla_root}/CarlaUE4.sh -carla-rpc-port={client_ports[i]} -nosound "
                            f"-carla-primary-port={carla_primary_ports[i]} -carla-streaming-port={sensor_ports[i]} "
                            f"-RenderOffScreen -graphicsadapter={graphics_adapter} -RPCThreads={tfv6_rpc_threads} "
                            f"-StreamingThreads={tfv6_streaming_threads} "
                            f"-SecondaryThreads={tfv6_secondary_threads}",
                            shell=True,
                            stdout=server_outs[i],
                            stderr=server_errs[i],
                        )
                    )
                else:
                    carla_processes.append(
                        subprocess.Popen(
                            # Keep CARLA server isolated from conda runtime libs to avoid UE4 lib conflicts.
                            f"env -u LD_LIBRARY_PATH "
                            f"bash {args.carla_root}/CarlaUE4.sh -carla-rpc-port={client_ports[i]} -nosound "
                            f"-carla-primary-port={carla_primary_ports[i]} -carla-streaming-port={sensor_ports[i]} "
                            f"-RenderOffScreen -graphicsadapter={graphics_adapter} -RPCThreads={tfv6_rpc_threads} "
                            f"-StreamingThreads={tfv6_streaming_threads} "
                            f"-SecondaryThreads={tfv6_secondary_threads}",
                            shell=True,
                            stdout=server_outs[i],
                            stderr=server_errs[i],
                        )
                    )
                time.sleep(server_sleep)

                print(f"Start client {i}")
                leaderboard_processes.append(
                    subprocess.Popen(
                        f"bash {args.repo_root}/rl_finetuning/start_leaderboard_tfv6_ppo.sh "
                        f"{args.repo_root} {args.git_root} {route_files[i]} {logdir} {i} "
                        f"{client_ports[i]} {traffic_manager_ports[i]} {rl_ports[i]} {args.seed} "
                        f"{skip_next_route} {args.route_repetitions} {args.tfv6_checkpoint} "
                        f"{args.track} {args.frame_rate} False {args.timeout} "
                        f"{args.runtime_timeout} {args.leaderboard_debug}",
                        shell=True,
                        stdout=client_outs[i],
                        stderr=client_errs[i],
                    )
                )
                time.sleep(client_sleep)

            skip_next_route = "False"

            cmdline = " ".join(map(shlex.quote, sys.argv[1:]))
            str_ports = " ".join(str(x) for x in rl_ports)

            load_file = None
            largest_step = 0
            if os.path.exists(logdir):
                for file in os.listdir(logdir):
                    if file.startswith("model_latest_") and file.endswith(".pth"):
                        full_path = os.path.join(logdir, file)
                        if os.path.getsize(full_path) > 0:
                            numbers_in_string = re.findall(r"\d+", file)
                            if numbers_in_string:
                                start_step = int(numbers_in_string[0])
                                if start_step > largest_step:
                                    largest_step = start_step
                                    load_file = os.path.join(logdir, file)

            if args.use_traj_sync_ppo:
                num_processes = args.num_envs_per_node
                num_envs_per_proc = 1
                print(f"Num processes : {num_processes}")
            else:
                num_processes = args.num_envs_per_node // args.num_envs_per_gpu
                num_envs_per_proc = args.num_envs_per_gpu
            world_size = num_processes * args.num_nodes
            if len(args.gpu_ids) < world_size:
                raise ValueError(
                    f"Need at least {world_size} gpu ids for trainer world size, got {len(args.gpu_ids)}."
                )

            if args.debug:
                train_out = open(
                    os.path.join(process_logdir, "train_out.txt"),
                    "w",
                    encoding="utf-8",
                )
                train_err = open(
                    os.path.join(process_logdir, "train_err.txt"),
                    "w",
                    encoding="utf-8",
                )
            else:
                train_out = sys.stdout
                train_err = sys.stderr

            train_process = subprocess.Popen(
                f"bash {args.repo_root}/rl_finetuning/start_learner_dd_ppo_tfv6_ppo.sh "
                f"{args.repo_root} {num_processes} {args.num_nodes} {args.rdzv_addr} {args.rdzv_port} "
                f"{cmdline} --ports {str_ports} --logdir {raw_logdir} --load_file {load_file} "
                f"--num_envs_per_proc {num_envs_per_proc} --tcp_store_port {tcp_store_port}",
                shell=True,
                stdout=train_out,
                stderr=train_err,
            )

            time.sleep(1)

            all_processes_running = True
            ended_leaderboard = []
            ended_carla = []
            for idx, _ in enumerate(carla_processes):
                ended_leaderboard.append(idx)
                ended_carla.append(idx)
            while all_processes_running:
                time.sleep(30)
                if train_process.poll() is not None:
                    all_processes_running = False
                    print("Train process ended")
                for _idx, carla_process in enumerate(carla_processes):
                    if carla_process.poll() is not None:
                        all_processes_running = False
                        print("Carla server crashed")
                        skip_next_route = "True"
                for _idx, leaderboard_process in enumerate(leaderboard_processes):
                    if leaderboard_process.poll() is not None:
                        all_processes_running = False
                        print("Leaderboard process ended")
                        skip_next_route = "True"

            for _ in range(360):
                for idx, carla_process in enumerate(carla_processes):
                    if carla_process.poll() is not None:
                        if idx in ended_carla:
                            ended_carla.remove(idx)
                            print(f"Server {idx} terminated")
                for idx, leaderboard_process in enumerate(leaderboard_processes):
                    if leaderboard_process.poll() is not None:
                        if idx in ended_leaderboard:
                            ended_leaderboard.remove(idx)
                time.sleep(1)

            for idx in ended_leaderboard:
                print(f"Leaderboard {idx} is hanging and did not terminate")

            print("Process finished:", train_process.returncode)
            if train_process.returncode == 0:
                print("Training finished succesfully")
                training = False

            cleanup(carla_processes, leaderboard_processes, train_process, client_ports)
            time.sleep(10)
            del carla_processes
            del leaderboard_processes
            del train_process

        blackhole.close()
        print("Finished cleanup")

    except KeyboardInterrupt:
        cleanup(carla_processes, leaderboard_processes, train_process, client_ports)
        blackhole.close()
        sys.exit(-1)
