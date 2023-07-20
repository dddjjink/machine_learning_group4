var websocket = new WebSocket('ws://127.0.0.1:8080')
  console.log(websocket.readyState)
  //readyState
  //0 链接没有建立(正在建立链接)
  //1 链接建立成功
  //2 链接正在关闭
  //3 链接已经关闭
  websocket.onopen = function () {
      console.log(websocket.readyState)
  }
  function send(){

    var dataset = document.querySelector('input[name="Dataset"]:checked').value;
    var splitter = document.querySelector('input[name="splitter"]:checked').value;
    var percent = document.querySelector('input[name="percent"]:checked');
    var model = document.querySelector('input[name="model"]:checked').value;
    var evaluation = document.querySelector('input[name="evaluation"]:checked').value;
    var data ={
      dataset:dataset,
      splitter:splitter,
      model:model,
      evaluation:evaluation
    }
    if (percent) {
  data.percent = percent.value;
    }
    websocket.send(JSON.stringify(data))
  }
  websocket.onmessage=function(back){
      console.log(back.data)
  }
