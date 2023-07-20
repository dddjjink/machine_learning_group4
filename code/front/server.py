import asyncio
import websockets
import json
from factory.Factory import DataFactory, EvaluationFactory, ModelFactory, SplitterFactory


# path='http://127.0.0.1:8080'
# 处理连接建立

# 定义处理函数
async def handle_message(websocket, path):
    async for message in websocket:
        try:
            while True:
                #text = ''
                #print(f"接收到的消息：{message}")
                #text += message
                #await websocket.send(message)
                message = eval(message)  # 加这一行，把str类型的message转换为dict类型
                dataset = message[list(message.keys())[0]]
                splitter = message[list(message.keys())[1]]
                model = message[list(message.keys())[2]]
                evaluation = message[list(message.keys())[3]]
                # percent = message[list(message.keys())[4]]      # 报错：list index out of range
                dataset_factory = DataFactory()
                _dataset_ = dataset_factory.create_dataset(dataset)
                _dataset_.data_target()
                # print(_dataset_.data)
                X = _dataset_.data
                y = _dataset_.target
                # print(X)
                splitter_factory = SplitterFactory()
                _splitter_ = splitter_factory.create_splitter(splitter, X, y)
                X_train, X_test, y_train, y_test = _splitter_.split()
                # print(X_train)
                model_factory = ModelFactory()
                _model_ = model_factory.create_model(model)
                _model_.fit(X_train)
                test_predict = _model_.predict(X_test)
                # print(test_predict)
                # print(y_test)
                evaluation_factory = EvaluationFactory()
                _evaluation_ = evaluation_factory.create_evaluation(evaluation, y_test, test_predict)

                result = _evaluation_()
                result_string=str(result)
                message["result"]=result_string
                print(message)
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


start_server = websockets.serve(handle_message, "localhost", 8081)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
