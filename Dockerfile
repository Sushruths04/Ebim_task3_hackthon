# EBiM Task 3 — Assisted Living & Feeding
#
# Builds from a published runtime image. The runtime contains the policy, its
# calibration and the arena map; this file pins the version, sets the defaults
# and declares the entrypoint, so `docker build .` from a clean checkout of the
# pinned commit reproduces exactly what we submit.
ARG RUNTIME=ghcr.io/sushruths04/ebim-task3-runtime:1.0.0
FROM ${RUNTIME}

LABEL org.opencontainers.image.title="EBiM Task 3 — Assisted Living & Feeding"
LABEL org.opencontainers.image.licenses="LicenseRef-Proprietary"

ENV ROS_DOMAIN_ID=0
ENTRYPOINT ["/usr/local/bin/ebim-run"]
CMD ["--object", "the rim of the cup", "--onto", "the rim of the plate"]
