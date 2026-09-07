# Thin layer on top of the vLLM image built from vllm's own docker/Dockerfile (target vllm-openai).
# Three things the cluster launcher (eugr's launch-cluster.sh) needs and that target lacks:
#   /workspace   : the launcher copies its mods and exec script there.
#   ray          : the launcher is Ray-based; the vllm-openai target does not ship it.
#   ENTRYPOINT [] : the target sets ENTRYPOINT ["vllm","serve"], which turns
#                  `docker run IMAGE python3 -c ...` (the launcher's memory probe) into
#                  an argument of `vllm serve`.
ARG BASE
FROM ${BASE}
RUN mkdir -p /workspace/mods && test -d /workspace && echo WORKSPACE_OK
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "ray[default]==2.56.1" \
 && ray --version && echo RAY_OK
ENTRYPOINT []
