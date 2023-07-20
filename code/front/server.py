import asyncio
import websockets
async def handle_connection(websocket,path):
    #path='http://127.0.0.1:8080'
    # 处理连接建立
    await websocket.send("连接成功，请发送数据集、测试集大小、算法和指标信息。")

    try:
        while True:
            message = await websocket.recv()
            data = message.split(",")
            dataset = data[0]
            splitter = data[1]
            model = data[2]
            evaluation = data[3]
            percent=data[4]

        dataset_factory=DatasetFactory()
        _dataset_=dataset_factory.create_dataset(dataset)




        splitter_factory = SplitterFactory()
        X_train,X_test,y_train,y_test = splitter_factory.create_splitter(splitter,X,y)

        model_factory = ModelFactory()
        _model_ = model_factory.create_model(model)


        evaluation_factory = EvaluationFactory()
        _evaluation_ = evaluation_factory.create_evaluation(evaluation,_data_)

        percent_factory = PercentFactory()
        _percent_ = percent_factory.create_percent(percent,_data_)

    except websockets.exceptions.ConnectionClosed:
        # 处理连接关闭
        print("客户端已断开连接")
start_server = websockets.serve(handle_connection, "localhost", 8080)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
