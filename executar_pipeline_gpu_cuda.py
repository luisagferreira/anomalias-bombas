"""
Orquestrador Completo para Execução em Servidor Remoto (SSH) com GPU CUDA.

Este script:
1. Verifica automaticamente a disponibilidade da GPU NVIDIA CUDA, versão do driver/CUDA e VRAM total.
2. Verifica se as planilhas de anomalias (`relatorio_anomalias_ano_completo_acima_2_5.csv`) e os
   tensores pareados 1D + 2D GAF (`.npz` em `Dataset_completo_bombas/Folds_Stratified_CV`) já existem
   no servidor SSH; caso você tenha transferido apenas os CSVs brutos (para economizar 15 GB de upload),
   ele gera automaticamente os folds estratificados, as matrizes GAF 32x32 e a planilha anual `>= 2.5`.
3. Executa o treinamento e avaliação Out-of-Fold (5 folds) das 7 arquiteturas na GPU usando CUDA e
   Automatic Mixed Precision (`torch.amp.autocast('cuda')`).

Exemplos de uso via SSH:
  # Execução padrão em GPU (20 épocas, batch_size=256, até 2000 normais + 2000 anômalas por config):
  python executar_pipeline_gpu_cuda.py

  # Execução em 100% do dataset (sem subamostragem) na GPU com 30 épocas:
  python executar_pipeline_gpu_cuda.py --full-dataset --epochs 30 --batch-size 512

  # Execução em segundo plano no SSH (sobrevive à desconexão do terminal):
  nohup python executar_pipeline_gpu_cuda.py --epochs 25 > log_gpu_cuda.txt 2>&1 &
"""

import os
import sys
import subprocess
import argparse
import torch


def check_cuda_environment():
    print("=" * 78, flush=True)
    print("DIAGNÓSTICO DE AMBIENTE GPU / CUDA (SERVIDOR SSH)", flush=True)
    print("=" * 78, flush=True)
    print(f"Python Version      : {sys.version.split()[0]}", flush=True)
    print(f"PyTorch Version     : {torch.__version__}", flush=True)
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA Disponível     : {cuda_ok}", flush=True)
    if cuda_ok:
        n_gpus = torch.cuda.device_count()
        print(f"Versão CUDA PyTorch : {torch.version.cuda}", flush=True)
        print(f"Quantidade de GPUs  : {n_gpus}", flush=True)
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024 ** 3)
            print(f"  -> GPU [{i}]        : {props.name} ({vram_gb:.2f} GB VRAM | Compute {props.major}.{props.minor})", flush=True)
    else:
        print("AVISO: PyTorch não detectou GPU CUDA ativa. Verifique `nvidia-smi` e a instalação do PyTorch com CUDA.", flush=True)
    print("=" * 78, flush=True)


def ensure_prerequisites():
    from aplicar_gaf_pareado import has_complete_gaf_npz

    # 1. Verifica se todos os folds CSV e arquivos .npz de GAF (inclusive Dataset_Completo) existem e contêm GASF/GADF/scalers
    sample_csv = os.path.join(
        "Dataset_completo_bombas",
        "Folds_Stratified_CV",
        "Dataset_Completo",
        "folds_Dataset_Completo_15min.csv",
    )
    sample_npz = os.path.join(
        "Dataset_completo_bombas",
        "Folds_Stratified_CV",
        "Bomba_1305",
        "sequencias_Bomba_1305_15min.npz",
    )
    if not os.path.isfile(sample_csv):
        print("\n[ETAPA PRÉVIA 1/2] Arquivos de Folds não encontrados localmente no servidor.", flush=True)
        print("Gerando Folds Estratificados (gerar_folds_estratificados.py)...", flush=True)
        subprocess.run([sys.executable, "gerar_folds_estratificados.py"], check=True)

    if not has_complete_gaf_npz(sample_npz):
        print("\n[ETAPA PRÉVIA 1/2] Tensores GADF/scalers ausentes nos .npz das bombas. Gerando GAF 32x32 (aplicar_gaf_pareado.py)...", flush=True)
        subprocess.run([sys.executable, "aplicar_gaf_pareado.py"], check=True)
    else:
        print("[OK] Tensores 1D + 2D GADF (.npz) das 6 bombas já presentes no disco.", flush=True)

    # 2. Verifica se a planilha anual StdDev >= 2.5 já existe
    annual_25_csv = os.path.join("Dataset_completo_bombas", "relatorio_anomalias_ano_completo_acima_2_5.csv")
    if not os.path.isfile(annual_25_csv):
        print("\n[ETAPA PRÉVIA 2/2] Gerando planilha anual completa e filtro StdDev >= 2.5...", flush=True)
        subprocess.run([sys.executable, "gerar_planilha_anual_acima_2_5.py"], check=True)
    else:
        print("[OK] Planilha anual StdDev >= 2.5 já presente no disco.", flush=True)


def main():
    has_cuda = torch.cuda.is_available()
    parser = argparse.ArgumentParser(description="Executa todo o projeto em GPU com CUDA via SSH")
    parser.add_argument("--device", type=str, default="cuda" if has_cuda else "cpu", help="Dispositivo ('cuda', 'cuda:0', 'cpu')")
    parser.add_argument("--epochs", type=int, default=20 if has_cuda else 7, help="Épocas de treinamento por fold")
    parser.add_argument("--batch-size", type=int, default=256 if has_cuda else 128, help="Batch size na GPU")
    parser.add_argument("--max-samples", type=int, default=2000 if has_cuda else 225, help="Amostras por classe por configuração")
    parser.add_argument("--full-dataset", action="store_true", help="Treina e avalia em 100%% de todas as janelas sem subamostragem")
    parser.add_argument("--no-resume", action="store_true", help="Recomeça do zero ignorando checkpoints anteriores")
    parser.add_argument("--also-run-reconstruction", action="store_true", help="Se ativado, roda também o benchmark puro de reconstrução (treinar_modelos_1d_2d_gaf.py)")
    args = parser.parse_args()

    check_cuda_environment()
    ensure_prerequisites()

    if args.also_run_reconstruction:
        print("\n>>> Executando Benchmark de Reconstrução 1D/2D (treinar_modelos_1d_2d_gaf.py)...", flush=True)
        subprocess.run([sys.executable, "treinar_modelos_1d_2d_gaf.py"], check=True)

    cmd = [
        sys.executable,
        "avaliar_modelos_planilha_anual_acima_2_5.py",
        "--device", args.device,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--max-samples", str(args.max_samples),
    ]
    if args.full_dataset:
        cmd.append("--full-dataset")
    if args.no_resume:
        cmd.append("--no-resume")

    print(f"\n>>> Executando Avaliação das 7 Arquiteturas em GPU ({' '.join(cmd)})...\n", flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
