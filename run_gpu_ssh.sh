#!/usr/bin/env bash
set -e

echo "============================================================"
echo " Verificando GPU NVIDIA (nvidia-smi)..."
echo "============================================================"
nvidia-smi || echo "Aviso: nvidia-smi nao encontrado no PATH."

echo ""
echo "============================================================"
echo " Iniciando Pipeline em GPU CUDA (Mixed Precision FP16)..."
echo "============================================================"
python3 executar_pipeline_gpu_cuda.py "$@"
