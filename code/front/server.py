import asyncio
import os
import webbrowser
import websockets
from factory.Factory import *

# path='http://127.0.0.1:8080'
# 处理连接建立

# 定义处理函数
async def handle_message(websocket, path):
    async for message in websocket:
        try:
            while True:

                if message == 'close_signal':
                    print("网页已关闭，连接已断开，进程终止")
                    # 收到关闭信号，断开连接并关闭程序
                    await websocket.close()
                    asyncio.get_event_loop().stop()  # 停止事件循环
                    return
                #text = ''
                #print(f"接收到的消息：{message}")
                #text += message
                #await websocket.send(message)
                message = eval(message)
                dataset = message.get("dataset")
                splitter = message.get("splitter")
                model = message.get("model")
                evaluation = message.get("evaluation")
                percent=message.get("percent")

                if dataset==None or splitter==None or model==None or evaluation==None or(splitter=="holdout" and percent==None):
                     await websocket.send("data_loss")

                dataset_factory = DataFactory()
                _dataset_ = dataset_factory.create_dataset(dataset)
                _dataset_.data_target()
                # print(_dataset_.data)
                X = _dataset_.data
                y = _dataset_.target
                # print(X)
                splitter_factory = SplitterFactory()
                _splitter_ = splitter_factory.create_splitter(splitter, X, y,percent)

                X_train, X_test, y_train, y_test = _splitter_.split()
                # print(X_train)
                model_factory = ModelFactory()
                _model_ = model_factory.create_model(model)
                if model == "KNN" or model == "K_means":
                    _model_.fit(X_train)
                else:
                    _model_.fit(X_train, y_train)
                test_predict = _model_.predict(X_test)
                # print(test_predict)
                # print(y_test)
                evaluation_factory = EvaluationFactory()
                _evaluation_ = evaluation_factory.create_evaluation(evaluation, y_test, test_predict)
                result = _evaluation_()
                result_string = str(result)
                if result == "curve":
                    message["result"] = result_string
                    await websocket.send(str(message))
                else:
                    message["result"] = result_string
                    print(f"接收到的消息：{message}")
                    await websocket.send(str(message))
                #message['result'] = result
                #print(message)
                #text += '\n' + str(message)
               # print(message)
                #await websocket.send(message)

                # response = f"数据集：{_dataset_.data}\n测试集大小：{test_size}\n算法：{algorithm}\n指标：{metric}\n得分：{score}"

            # await websocket.send(message)

        except websockets.exceptions.ConnectionClosed:
            # 处理连接关闭

            print("客户端已断开连接")

def run():
    start_server = websockets.serve(handle_message, "localhost", 8082)
    # 启动服务器
    asyncio.get_event_loop().run_until_complete(start_server)
    # 打开网页
    path = os.path.join(os.path.dirname(__file__), 'web.html')
    webbrowser.open(path)
    # 运行事件循环
    asyncio.get_event_loop().run_forever()

