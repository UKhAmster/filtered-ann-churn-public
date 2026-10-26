# ACORN — форк FAISS, коммит закреплён по SHA: репозиторий заморожен с марта 2025,
# и ветка при этом всё равно может уехать.
FROM ubuntu:22.04
ARG REPO_URL=https://github.com/guestrin-lab/ACORN.git
ARG COMMIT_SHA=c259f11cdbec880671658eaa0d8747e2cc9de79d
ENV DEBIAN_FRONTEND=noninteractive
# zlib нужен FAISS для сериализации индексов; без него cmake падает на find_package(ZLIB).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake libomp-dev libopenblas-dev zlib1g-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone "$REPO_URL" /src && cd /src && git checkout "$COMMIT_SHA"
WORKDIR /src
# Два отступления от README проекта, оба вынужденные.
#
# BUILD_TESTING=OFF: тесты FAISS к ACORN отношения не имеют, а demos (где лежит
# test_acorn) подключаются независимо от этого флага.
#
# CMAKE_SKIP_INSTALL_RULES=ON: корневой CMakeLists ACORN тянет nlohmann_json через
# FetchContent и добавляет его путь в цель faiss как PUBLIC, без обёртки
# $<BUILD_INTERFACE:>. При генерации export-набора CMake отвергает такой путь
# («prefixed in the build directory») и обрывает конфигурацию. Устанавливать
# библиотеку в систему нам не нужно — отключение install-правил снимает проблему,
# не трогая исходники метода.
RUN cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF \
          -DCMAKE_SKIP_INSTALL_RULES=ON \
          -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -B build
RUN make -C build -j"$(nproc)" faiss
# Собственный тест проекта: собирается ради проверки компиляции, но не запускается —
# ему нужен внешний датасет SIFT с готовыми метаданными и эталонами.
RUN make -C build -j"$(nproc)" test_acorn
COPY smoke_acorn.cpp /smoke/smoke_acorn.cpp
RUN g++ -O2 -std=c++17 -fopenmp -I/src /smoke/smoke_acorn.cpp \
        -L/src/build/faiss -lfaiss -o /smoke/smoke_acorn
ENV LD_LIBRARY_PATH=/src/build/faiss
CMD ["/smoke/smoke_acorn"]
