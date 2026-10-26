# UNG. В репозитории единственный коммит `init` от декабря 2024, истории нет,
# откатываться при поломке некуда — ревизия закреплена по SHA как есть.
FROM ubuntu:22.04
ARG REPO_URL=https://github.com/YZ-Cai/Unified-Navigating-Graph.git
ARG COMMIT_SHA=92b58ffd600d3a51bb6537460dae1079150b5fa5
ENV DEBIAN_FRONTEND=noninteractive
# MKL не нужен: codes/CMakeLists.txt требует только Boost и OpenMP, а пакет
# libmkl-full-dev к тому же задаёт интерактивный вопрос и вешает сборку.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake ca-certificates \
        libboost-program-options-dev libboost-filesystem-dev \
        libomp-dev libaio-dev libgoogle-perftools-dev \
        python3 python3-numpy \
    && rm -rf /var/lib/apt/lists/*
RUN git clone "$REPO_URL" /src && cd /src && git checkout "$COMMIT_SHA"
WORKDIR /src
RUN mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release ../codes/ && make -j"$(nproc)"
COPY smoke.sh /smoke.sh
CMD ["bash", "/smoke.sh"]
