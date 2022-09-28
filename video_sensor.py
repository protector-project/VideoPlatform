from fileinput import filename
from dagster import sensor, RunRequest, DefaultSensorStatus, repository
from minio import Minio
from minio.commonconfig import Tags
from video_processing_pipeline import processing_video_pipeline
import os

@sensor(job=processing_video_pipeline)
def my_bucket_sensor(default_status=DefaultSensorStatus.RUNNING):
    envs = {
        "MINIO_ADDRESS": os.environ.get("MINIO_ADDRESS"),
        "MINIO_ACCESS_KEY": os.environ.get("MINIO_ACCESS_KEY"),
        "MINIO_SECRET_KEY": os.environ.get("MINIO_SECRET_KEY"),
        "RAW_BUCKET": os.environ.get("RAW_BUCKET"),
        "VIDEO_BUCKET": os.environ.get("VIDEO_BUCKET"),
        "INFLUX_BUCKET": os.environ.get("INFLUX_BUCKET"),
        "OUT_FOLDER": os.environ.get("OUT_FOLDER")
    }

    client = Minio(
        envs["MINIO_ADDRESS"],
        access_key=envs["MINIO_ACCESS_KEY"],
        secret_key=envs["MINIO_SECRET_KEY"],
        secure=False,
    )
    bucket_name = envs["RAW_BUCKET"]
    response = client.list_objects(bucket_name, recursive=True)
    for resp in response:

        tags = client.get_object_tags(bucket_name, resp.object_name)

        if tags == None or tags["status"] != "processed":
            # create run
            yield RunRequest(
                run_key=resp.object_name,
                run_config={
                    "ops": {
                        "load_envs": {
                            "config": {
                                "MINIO_ADDRESS": envs["MINIO_ADDRESS"],
                                "MINIO_ACCESS_KEY": envs["MINIO_ACCESS_KEY"],
                                "MINIO_SECRET_KEY": envs["MINIO_SECRET_KEY"],
                                "RAW_BUCKET": envs["RAW_BUCKET"],
                                "VIDEO_BUCKET": envs["VIDEO_BUCKET"],
                                "INFLUX_BUCKET": envs["INFLUX_BUCKET"],
                                "FILE_NAME": resp.object_name,
                                "OUT_FOLDER": envs["OUT_FOLDER"]
                            }
                        }
                    }
                },
            )

            if tags == None:
                tags = Tags.new_object_tags()
            tags["status"] = "processed"
            client.set_object_tags(bucket_name, resp.object_name, tags)


@repository
def my_repository():
    return [processing_video_pipeline, my_bucket_sensor]
