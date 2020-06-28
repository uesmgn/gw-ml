echo $(pwd)
python -m torch_xla.distributed.xla_dist \
      --tpu=$TPU_NAME \
      --conda-env=torch-xla-nightly \
      --env XLA_USE_BF16=1 \
      python $(pwd)/search_cvae.py --use_tpu \
