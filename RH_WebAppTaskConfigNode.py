class WebAppTaskConfigNode:
    """
    WebApp任务配置节点 - 为 nodeInfoList 添加 is_webapp_task 和 workflowId_webappId 配置信息
    可以连接到 NodeInfoListNode 的输出，添加 WebApp 任务标记和工作流ID
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "is_webapp_task": ("BOOLEAN", {"default": False, "tooltip": "是否为WebApp任务"}),
                "workflowId_webappId": ("STRING", {"default": "", "tooltip": "工作流ID或WebApp ID"}),
            },
            "optional": {
                "nodeInfoList": ("ARRAY", {"default": [], "tooltip": "节点信息列表（来自 NodeInfoListNode）"}),
            }
        }
    
    RETURN_TYPES = ("ARRAY",)
    RETURN_NAMES = ("nodeInfoList",)
    CATEGORY = "RunningHub"
    FUNCTION = "process"
    OUTPUT_NODE = False
    
    def process(self, is_webapp_task, workflowId_webappId, nodeInfoList=None):
        """
        为 nodeInfoList 添加 is_webapp_task 和 workflowId_webappId 配置信息
        使用特殊的元数据项来标记：
        - nodeId=-1, fieldName="_is_webapp_task", fieldValue="true"/"false"
        - nodeId=-2, fieldName="_workflowId_webappId", fieldValue=workflowId_webappId
        """
        result_list = []
        
        # 如果提供了 nodeInfoList，先复制它
        if nodeInfoList:
            # nodeInfoList 可能是嵌套列表（ComfyUI 的 ARRAY 类型）
            if isinstance(nodeInfoList, list) and len(nodeInfoList) > 0:
                # 检查是否是嵌套列表
                if isinstance(nodeInfoList[0], list):
                    # 如果是嵌套列表，取第一个元素
                    result_list = nodeInfoList[0].copy()
                else:
                    # 如果不是嵌套列表，直接使用
                    result_list = nodeInfoList.copy()
        
        # 移除旧的 _is_webapp_task 标记（如果存在）
        result_list = [item for item in result_list if not (
            isinstance(item, dict) and 
            item.get("nodeId") == -1 and 
            item.get("fieldName") == "_is_webapp_task"
        )]
        
        # 移除旧的 _workflowId_webappId 标记（如果存在）
        result_list = [item for item in result_list if not (
            isinstance(item, dict) and 
            item.get("nodeId") == -2 and 
            item.get("fieldName") == "_workflowId_webappId"
        )]
        
        # 添加新的 _is_webapp_task 标记
        webapp_config = {
            "nodeId": -1,  # 使用 -1 作为特殊标记
            "fieldName": "_is_webapp_task",  # 使用特殊字段名
            "fieldValue": "true" if is_webapp_task else "false"
        }
        result_list.append(webapp_config)
        
        # 添加新的 _workflowId_webappId 标记
        if workflowId_webappId:
            workflow_config = {
                "nodeId": -2,  # 使用 -2 作为特殊标记
                "fieldName": "_workflowId_webappId",  # 使用特殊字段名
                "fieldValue": str(workflowId_webappId)
            }
            result_list.append(workflow_config)
        
        print(f"WebAppTaskConfigNode: Added is_webapp_task={is_webapp_task}, workflowId_webappId={workflowId_webappId} to nodeInfoList")
        
        # 返回嵌套列表格式（与 NodeInfoListNode 保持一致）
        return [result_list]

