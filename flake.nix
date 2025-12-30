{
  description = "39C3 Transformer Workshop";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nix-ai.url = "git+https://git.wavelens.io/public/nix-ai";
    nix-ai.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { nix-ai, ... }: nix-ai.lib.mkFlake {
    presets = {
      datascience = true;
      jupyter = true;
      torch = true;
      huggingface = true;
    };

    pythonPackages = [ "einops" ];
    datasets = {
      addition10.prepare = {
        directoryPath = ./datasets/addition;
        drop = [
          "dataset_train.pt"
          "dataset_test.pt"
        ];

        commands = ''
          jupyter nbconvert --to script generate.ipynb
          python3 generate.py --max-number 10 --train-test-split 0.5
        '';
      };
    };

    trainings = {
      # hubert = {
      #   GPU = "CPU";
      #   directoryPath = ./models/hubert;
      #   mergeDirectories = [ "runs" ];
      #   copyDatasets = [ "librispeechStatic" "librispeechStaticTest" ];

      #   commands = ''
      #     pytorch train.py
      #   '';

      #   drop = [
      #     "transformer.pt"
      #     "runs"
      #   ];

      #   configurations = [
      #     {
      #       file = "transformer.pt";
      #     }
      #   ];

      #   testConfiguration = {
      #     file = "transformer.pt";
      #   };
      # };
    };
  };
}
