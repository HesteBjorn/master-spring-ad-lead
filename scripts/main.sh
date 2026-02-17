# CARLA environment variables
export CARLA_VERSION="0915"
export CARLA_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/CARLA_${CARLA_VERSION}"

# Python paths
export PYTHONPATH=${CARLA_ROOT}/PythonAPI/carla:${PYTHONPATH}
export PYTHONPATH=${CARLA_ROOT}/PythonAPI:${PYTHONPATH}

# CaRL custom leaderboard paths (used by rl_finetuning/tfv6 PPO parallel training)
export CARL_CARLA_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/CaRL/CARLA"
if [ -d "${CARL_CARLA_ROOT}" ]; then
  export WORK_DIR="${CARL_CARLA_ROOT}"
  export SCENARIO_RUNNER_ROOT="${WORK_DIR}/custom_leaderboard/scenario_runner"
  export LEADERBOARD_ROOT="${WORK_DIR}/custom_leaderboard/leaderboard"
  export PYTHONPATH="${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}:${PYTHONPATH}"
fi

# System paths
export PATH=$LEAD_PROJECT_ROOT:$PATH
export PATH=$LEAD_PROJECT_ROOT/scripts:$PATH

# NavSim
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/navsimv1.1"
export OPENSCENE_DATA_ROOT="${LEAD_PROJECT_ROOT}/3rd_party/navsim_workspace/dataset"
