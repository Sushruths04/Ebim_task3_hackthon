# EBiM Task 3 — Assisted Living & Feeding
#
# Builds from the published mission image, which carries the whole policy for
# linux/amd64 and linux/arm64 (the robot's companion is arm64). This file pins
# the version, sets the defaults and declares the entrypoint, so `docker build .`
# reproduces what we run. The run commands are in README.md; on the Jetson,
# `docker build` and `docker run` both need `--network host`.
ARG RUNTIME=ghcr.io/sushruths04/ebim-task3-mission:1.4.1
FROM ${RUNTIME}

LABEL org.opencontainers.image.title="EBiM Task 3 — Assisted Living & Feeding"
LABEL org.opencontainers.image.licenses="LicenseRef-Proprietary"

ENV ROS_DOMAIN_ID=0
ENTRYPOINT ["/usr/local/bin/ebim-run"]
CMD ["mission", "--stage", "all"]
