"""
this script copies easyocr checkpoints to ~/.EasyOCR/model and prepares the config file
"""
import argparse
import os
import shutil
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        __doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("model_path", type=str, help="Path to model")
    parser.add_argument(
        "--output_name",
        type=str,
        default="{exp_name}_{model_name}",
        help="Path to model",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_files/{exp_name}.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--pyfile", type=str, default="../custom_example.py", help="Path to config file"
    )
    parser.add_argument("--dry_run", action="store_true", help="Dry run")
    parser.add_argument("--no_test", action="store_true", help="don't test. Test will try to load the OCR")
    args = parser.parse_args()

    os.makedirs(os.path.expanduser("~/.EasyOCR/model"), exist_ok=True)
    os.makedirs(os.path.expanduser("~/.EasyOCR/user_network"), exist_ok=True)

    exp_name = os.path.basename(os.path.split(args.model_path.rstrip("/"))[0])
    model_name = os.path.basename(
        os.path.split(args.model_path.rstrip("/"))[1]
    ).replace(".pth", "")
    args.config = args.config.format(exp_name=exp_name, model_name=model_name)
    args.output_name = args.output_name.format(exp_name=exp_name, model_name=model_name)

    print(
        "copying",
        args.model_path,
        os.path.expanduser(f"~/.EasyOCR/model/{args.output_name}.pth"),
    )
    print(
        "copying",
        args.config,
        os.path.expanduser(f"~/.EasyOCR/user_network/{args.output_name}.yaml"),
    )
    print(
        "copying",
        args.pyfile,
        os.path.expanduser(f"~/.EasyOCR/user_network/{args.output_name}.py"),
    )

    if not args.dry_run:
        shutil.copy(
            args.model_path,
            os.path.expanduser(f"~/.EasyOCR/model/{args.output_name}.pth"),
        )
        shutil.copy(
            args.pyfile,
            os.path.expanduser(f"~/.EasyOCR/user_network/{args.output_name}.py"),
        )

        opt = yaml.safe_load(open(args.config, encoding='utf8'))
        opt["character"] = opt.get("character", opt["number"] + opt["symbol"] + opt["lang_char"])
        opt["character_list"] = opt["character"]
        print('opt["character_list"]', len(opt["character_list"]), opt["character_list"])
        opt["network_params"] = {
            "input_channel": opt["input_channel"],
            "output_channel": opt["output_channel"],
            "hidden_size": opt["hidden_size"],
        }
        opt["imgH"] = opt["imgH"]
        opt["lang_list"] = [opt.get("lang_list", "ar")]
        # write to yaml
        with open(
            os.path.expanduser(f"~/.EasyOCR/user_network/{args.output_name}.yaml"), "w"
        ) as f:
            yaml.dump(opt, f)
        # shutil.copy(args.config, os.path.expanduser(f'~/.EasyOCR/user_network/{args.output_name}.yaml'))

        if not args.no_test:
            print('testing model')
            import easyocr

            reader = easyocr.Reader(["ar"], recog_network=args.output_name)

        print(
            f"Done. Now you can use the model by calling: "
            f"easyocr.Reader(['ar'], recog_network='{args.output_name}')"
        )
