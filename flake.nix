{
	description = "KeyZox's neovim config";
	inputs = {
		nixpkgs.url = "github:nixos/nixpkgs/nixos-24.05";
		flake-utils.url = "github:numtide/flake-utils";
	};

	outputs = { nixpkgs, flake-utils, ... }: flake-utils.lib.eachDefaultSystem (system:
		let 
			pkgs = nixpkgs.legacyPackages.${system};
		in
		{
			devShells = ({
			    default = pkgs.mkShell.override
				{
				# Override stdenv in order to change compiler:
				# stdenv = pkgs.clangStdenv;
				}
				{
					buildInputs = with pkgs; [
						llvmPackages.openmp
					];

					LD_LIBRARY_PATH="${pkgs.vulkan-loader}/lib";

					packages = with pkgs; [
						clang-tools
						clang
						gcc
					];
				};
			});
    });
}
