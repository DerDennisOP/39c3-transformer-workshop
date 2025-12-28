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

    pythonPackages = [
      "termcolor"
      "tqdm"
      "matplotlib"
      "einops"
      "zstandard"
    ];

    datasets = {
      # librispeechStatic = {
      #   src = {
      #     url = "https://static.wavelens.io/private/corpusP.zip";
      #     hash = "sha256-hGqdUD5/OuhCTla338M30ggGOCTGIprjdbdVAAqdK9o=";
      #   };

      #   prepare = {
      #     drop = [ "." ];
      #     commands = ''
      #       unzip corpusP.zip
      #     '';
      #   };
      # };
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
