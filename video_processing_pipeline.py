import pandas as pd
from dagster import get_dagster_logger, job, op
from minio import Minio
import os
from io import BytesIO
from subprocess import PIPE, Popen
import shutil


@op(
    config_schema={
        "MINIO_ADDRESS": str,
        "MINIO_ACCESS_KEY": str,
        "MINIO_SECRET_KEY": str,
        "RAW_BUCKET": str,
        "VIDEO_BUCKET": str,
        "INFLUX_BUCKET": str,
        "FILE_NAME": str,
        "OUT_FOLDER": str,
    }
)
def load_envs(context) -> dict:
    config = {
        "MINIO_ADDRESS": context.op_config["MINIO_ADDRESS"],
        "MINIO_ACCESS_KEY": context.op_config["MINIO_ACCESS_KEY"],
        "MINIO_SECRET_KEY": context.op_config["MINIO_SECRET_KEY"],
        "RAW_BUCKET": context.op_config["RAW_BUCKET"],
        "VIDEO_BUCKET": context.op_config["VIDEO_BUCKET"],
        "INFLUX_BUCKET": context.op_config["INFLUX_BUCKET"],
        "FILE_NAME": context.op_config["FILE_NAME"],
        "OUT_FOLDER": context.op_config["OUT_FOLDER"],
    }
    return config


@op
def download_raw_data(env_values: dict):
    client = Minio(
        env_values["MINIO_ADDRESS"],
        access_key=env_values["MINIO_ACCESS_KEY"],
        secret_key=env_values["MINIO_SECRET_KEY"],
        secure=False,
    )
    bucket_name = env_values["RAW_BUCKET"]
    input_name = env_values["FILE_NAME"]
    obj = client.fget_object(
        bucket_name,
        input_name,
        input_name,
    )

    return input_name


@op
def convert_to_mp4(input_folder: str):
    logger = get_dagster_logger()
    logger.info("Input folder: {}".format(input_folder))
    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)
        logger.info("FILE: {}".format(filepath))
        if os.path.isfile(filepath):
            if os.path.splitext(filename)[1] == ".avi":
                # convert file to mp4
                c = "ffmpeg -i {} -y {}.mp4".format(filepath, os.path.splitext(filepath)[0])
                logger.info("COMMAND: {}".format(c))
                c_ = c.split(" ")
                process = Popen(c_, stdout=PIPE, stderr=PIPE, shell=False)
                stdout, stderr = process.communicate()
                logger.info(str(stdout))
                logger.info(str(stderr))


    return "done"


@op
def upload_out_folder(context, env_values: dict, out_dir: str, status: str):
    client = Minio(
        env_values["MINIO_ADDRESS"],
        access_key=env_values["MINIO_ACCESS_KEY"],
        secret_key=env_values["MINIO_SECRET_KEY"],
        secure=False,
    )
    for filename in os.listdir(out_dir):
        filepath = os.path.join(out_dir, filename)
        if os.path.isfile(filepath):
            context.log.info(filename)
            with open(filepath, "rb") as file_bytes:
                b_data = file_bytes.read()
                stream_input_file = BytesIO(b_data)
                if os.path.splitext(filename)[1] == ".mp4":
                    client.put_object(
                        env_values["VIDEO_BUCKET"],
                        filename,
                        data=stream_input_file,
                        length=len(b_data),
                        content_type="application/mp4",
                    )
                elif os.path.splitext(filename)[1] == ".json":
                    client.put_object(
                        env_values["INFLUX_BUCKET"],
                        filename,
                        data=stream_input_file,
                        length=len(b_data),
                        content_type="application/json",
                    )
        os.remove(filepath)

    return env_values["FILE_NAME"]


@op
def detect_configuration(env_values: dict, input_video: str) -> dict:
    config = {}
    split_file_name = input_video.split(".")[0]
    if "piazza-2-sett" in split_file_name:
        config["input_name"] = "piazza-2-sett"
        config["config_file"] = "config/mt.yaml"
        config["output_root"] = env_values["OUT_FOLDER"]
    else:
        config = None
    return config


@op
def video_processing(input_video: str, run_config: dict) -> str:
    c = "python src/main.py --input_video {} --input_name {} --config_file {} --output_root {}".format(
        input_video, run_config["input_name"], run_config["config_file"], run_config["output_root"]
    )
    c_ = c.split(" ")
    process = Popen(c_, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    process.kill()
    return str(run_config["output_root"])


@op
def delete_input_file(input_file: str):
    os.remove(input_file)


@job
def processing_video_pipeline():
    env_values = load_envs()
    input_name = download_raw_data(env_values)
    configuration = detect_configuration(env_values, input_name)
    result = video_processing(input_name, configuration)
    status = convert_to_mp4(result)
    result = upload_out_folder(env_values, result, status)
    delete_input_file(result)
