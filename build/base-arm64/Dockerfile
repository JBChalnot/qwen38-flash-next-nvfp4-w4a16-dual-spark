# Build base for vLLM on GB10 / aarch64 / CUDA 13.
#
# vLLM's docker/Dockerfile expects `pytorch/manylinux2_28-builder:cuda13.0`, which has no arm64
# manifest. This image imitates that manylinux environment on `nvidia/cuda:13.0.3-devel-rockylinux9`
# and is passed as `--build-arg BUILD_BASE_IMAGE=...`. It does not modify the upstream Dockerfile.
# Each package below fixes a build failure that was actually observed:
#   EPEL            -> "No match for argument: ccache"
#   full curl       -> "curl-minimal ... conflicts with curl"
#   /opt/python/... -> "No interpreter found at /opt/python/cp312-cp312/bin/python3.12"
#   patch etc.      -> CMakeLists.txt:417 execute_process(COMMAND patch ...)
#   openssl-devel   -> build_rust.sh: "failed to run custom build command for openssl-sys"
FROM nvidia/cuda:13.0.3-devel-rockylinux9
ARG PY=3.12
# The CUDA repo metadata inside this image is stale (404 on repodata), which fails EVERY dnf
# transaction. Nothing is needed from it: the toolkit is already installed.
RUN dnf clean all \
 && dnf --disablerepo=cuda install -y epel-release \
 && dnf --disablerepo=cuda install -y --allowerasing curl \
 && dnf --disablerepo=cuda install -y \
      python${PY} python${PY}-devel python${PY}-pip \
      patch which diffutils file findutils \
      gcc gcc-c++ make cmake \
      git ccache sudo rdma-core-devel \
      xz bzip2 zip unzip tar gzip \
      openssl-devel perl perl-FindBin perl-IPC-Cmd pkgconf-pkg-config \
      libffi-devel zlib-devel \
 && dnf clean all && rm -rf /var/cache/dnf
# Disable the CUDA repo durably: the upstream Dockerfile runs its own `dnf install` later.
RUN for f in /etc/yum.repos.d/*cuda*.repo; do \
      [ -f "$f" ] && sed -i 's/^enabled=1/enabled=0/' "$f"; done; true
# The manylinux python layout the upstream Dockerfile hard-codes.
RUN PYV_NODOT=$(echo ${PY} | tr -d '.') \
 && mkdir -p /opt/python/cp${PYV_NODOT}-cp${PYV_NODOT}/bin \
 && ln -sf /usr/bin/python${PY} /opt/python/cp${PYV_NODOT}-cp${PYV_NODOT}/bin/python${PY} \
 && ln -sf /usr/bin/python${PY} /opt/python/cp${PYV_NODOT}-cp${PYV_NODOT}/bin/python3
# Fail closed here rather than ten minutes into the compile.
RUN for b in patch which diff file find gcc g++ make cmake git ccache tar xz perl pkg-config; do \
      command -v $b >/dev/null || { echo "MISSING: $b"; exit 1; }; done \
 && test -f /usr/include/openssl/ssl.h || { echo 'MISSING: openssl headers'; exit 1; } \
 && /opt/python/cp312-cp312/bin/python3.12 --version \
 && echo BASE_ARM64_OK
