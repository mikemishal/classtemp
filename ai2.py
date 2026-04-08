import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

def get_image_caption(image_url):
    try:
        endpoint = os.environ["VISION_ENDPOINT"]
        key = os.environ["VISION_KEY"]
    except KeyError:
        print("Missing environment variable 'VISION_ENDPOINT' or 'VISION_KEY'")
        print("Set them before running this function.")
        return None

    # Create an Image Analysis client
    client = ImageAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )

    # Get a caption for the image
    result = client.analyze_from_url(
        image_url=image_url,
        visual_features=[VisualFeatures.CAPTION],
        gender_neutral_caption=True  # Optional (default is False)
    )

    if result.caption is not None:
        return f"'{result.caption.text}', Confidence {result.caption.confidence:.4f}"
    else:
        return "No caption available."