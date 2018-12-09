from nmt.nmt import *
# tf.enable_eager_execution()


def main(unused_argv):
  default_hparams = create_hparams(FLAGS)
  train_fn = train.train
  inference_fn = inference.inference
  run_main(FLAGS, default_hparams, train_fn, inference_fn)


if __name__ == "__main__":
  nmt_parser = argparse.ArgumentParser()
  add_arguments(nmt_parser)
  FLAGS, unparsed = nmt_parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)