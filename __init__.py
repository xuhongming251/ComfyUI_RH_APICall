from .RH_SettingsNode import SettingsNode
from .RH_BaseSettingsNode import BaseSettingsNode
from .RH_NodeInfoListNode import NodeInfoListNode
from .RH_ExecuteNode import ExecuteNode
from .RH_ImageUploaderNode import ImageUploaderNode
from .RH_VideoUploader import RH_VideoUploader
from .RH_AudioUploader import RH_AudioUploader
from .RH_AsyncExecuteNode import AsyncBatchExecuteNode, AsyncExtractResultNode
from .RH_WebAppTaskConfigNode import WebAppTaskConfigNode

from .RH_Utils import *



NODE_CLASS_MAPPINGS = {
    "RH_SettingsNode": SettingsNode,
    "RH_BaseSettingsNode": BaseSettingsNode,
    "RH_NodeInfoListNode": NodeInfoListNode,
    "RH_ExecuteNode": ExecuteNode,
    "RH_ImageUploaderNode": ImageUploaderNode,
    "RH_Utils": AnyToStringNode,
    "RH_ExtractImage": RH_Extract_Image_From_List,
    "RH_BatchImages": RH_Batch_Images_From_List,
    "RH_VideoUploader": RH_VideoUploader,
    "RH_AudioUploader": RH_AudioUploader,
    "RH_AsyncBatchExecuteNode": AsyncBatchExecuteNode,
    "RH_AsyncExtractResultNode": AsyncExtractResultNode,
    "RH_WebAppTaskConfigNode": WebAppTaskConfigNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RH_SettingsNode": "RH Settings",
    "RH_BaseSettingsNode": "RH Base Settings",
    "RH_NodeInfoListNode": "RH Node Info List",
    "RH_ExecuteNode": "RH Execute",
    "RH_ImageUploaderNode": "RH Image Uploader",
    "RH_Utils": "RH Anything to String",
    "RH_ExtractImage": "RH Extract Image From ImageList",
    "RH_BatchImages": "RH Batch Images From ImageList",
    "RH_VideoUploader": "RH Video Uploader",
    "RH_AudioUploader": "RH Audio Uploader",
    "RH_AsyncBatchExecuteNode": "RH Async Batch Execute",
    "RH_AsyncExtractResultNode": "RH Async Extract Result",
    "RH_WebAppTaskConfigNode": "RH WebApp Task Config",
}

# Web Directory Setup
# Tells ComfyUI where to find the JavaScript files associated with nodes in this package
WEB_DIRECTORY = "./web/js"


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
