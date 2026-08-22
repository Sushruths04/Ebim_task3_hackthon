# EBiM Task 3 submission image -- the ONE build story (REV12 T2, 2026-08-06).
#
# Historically this repo had two competing Dockerfiles: this file (a dev
# shim whose entrypoint called scripts/task3/run_episode.py's CLI, which
# raises NotImplementedError for anything but policy="idle" -- see
# run_episode.py:224-227) and docker/Dockerfile.submission (the real,
# GPU-verified submission recipe, entrypoint docker/
# task3_submission_entrypoint.sh -> task3_pipeline.run_task3, the actual
# 4-stage orchestrator). An evaluator running the documented default
# `docker build .` at the repo root got the broken one. This file now IS
# that real recipe -- docker/Dockerfile.submission is kept as an
# identical copy for any tooling that references it by path explicitly,
# but the root Dockerfile is the canonical, documented entrypoint.
#
# Base image pinned by digest (not just tag) so a future NGC repush of
# the `2.3.2` tag cannot silently change what gets built. Digest
# confirmed 2026-08-06 via `docker manifest inspect` and the registry's
# own `Docker-Content-Digest` header, both in agreement. No `docker
# login` is required to pull it -- verified anonymously pullable,
# docker/SUBMISSION_README.md.
ARG BASE_IMAGE=nvcr.io/nvidia/isaac-lab@sha256:f07c37e3f0c9f58f7febd0aa9a425523d282be623c0db81ac61006d0e24be07f
FROM ${BASE_IMAGE}

ARG WORKSPACE_ROOT=/workspace/EBiM_Challenge
WORKDIR ${WORKSPACE_ROOT}

ENV ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    PRIVACY_USERID=submission \
    OMNI_KIT_ALLOW_ROOT=1 \
    HOME=/root

# Inference only -- never ship the trainer. Build context is the repo
# root; the repo's own root .dockerignore (.git/, docs/, outputs/, logs,
# recordings, __pycache__) applies.
COPY . ${WORKSPACE_ROOT}

RUN chmod +x docker/task3_submission_entrypoint.sh

# Installs cuRobo (batch inverse kinematics) into the image at build time,
# pinned to a specific NVLabs/curobo commit (see the script) so the base
# image's IK backend is reproducible, not whatever HEAD happens to be the
# day of the build.
RUN bash scripts/task3/rebuild_curobo_env.sh

ENTRYPOINT ["/workspace/EBiM_Challenge/docker/task3_submission_entrypoint.sh"]
