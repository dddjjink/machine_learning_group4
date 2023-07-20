function updateOptions() {
    var dataset = document.querySelector('input[name="Dataset"]:checked').value;

    var distanceOption = document.getElementById("distanceOption");
    distanceOption.style.display = (dataset === "iris" ) ? "none" : "inline";

    var mseOption = document.getElementById("mseOption");
    mseOption.style.display = (dataset === "iris" ) ? "none" : "inline";

    var rmseOption = document.getElementById("rmseOption");
    rmseOption.style.display = (dataset === "iris" ) ? "none" : "inline";

    var aucOption = document.getElementById("aucOption");
    aucOption.style.display = (dataset === "heart") ? "inline" : "none";

    var f1Option = document.getElementById("f1Option");
    f1Option.style.display = (dataset === "heart") ? "inline" : "none";

    var prOption = document.getElementById("prOption");
    prOption.style.display = (dataset === "heart") ? "inline" : "none";


    var rocOption = document.getElementById("rocOption");
    rocOption.style.display = (dataset === "heart") ? "inline" : "none";



    var GradientBoostingOption = document.getElementById("GradientBoostingOption");
    GradientBoostingOption.style.display = (dataset === "iris") ? "none" : "inline";

    var LROption = document.getElementById("LROption");
    LROption.style.display = (dataset === "iris") ? "none" : "inline";

    var LogisticRegressionOption = document.getElementById("LogisticRegressionOption");
    LogisticRegressionOption.style.display = (dataset === "heart") ? "inline" : "none";

    var SVMOption = document.getElementById("SVMOption");
    SVMOption.style.display = (dataset === "heart") ? "inline" : "none";

    var DecisionTreeOption = document.getElementById("DecisionTreeOption");
    DecisionTreeOption.style.display = (dataset === "iris") ? "none" : "inline";

     var RandomForestOption = document.getElementById("RandomForestOption");
    RandomForestOption.style.display = (dataset === "iris") ? "none" : "inline";

}
function updatepercent() {
    var splitter = document.querySelector('input[name="splitter"]:checked').value;

    var percentOption = document.getElementById("percentOption");
    percentOption.style.display = (splitter === "houldout") ? "block" : "none";
}
function updatedata() {
    var model = document.querySelector('input[name="model"]:checked').value;

    var irisOption = document.getElementById("irisOption");
    irisOption.style.display = (model === "K_means" || model === "KNN" || model === "NB") ? "inline" : "none";

    var wineOption = document.getElementById("wineOption");
    wineOption.style.display = (model === "SVM" || model === "LogisticRegression") ? "none" : "inline";

}
function updatedata1() {
    var evaluation = document.querySelector('input[name="evaluation"]:checked').value;

    var irisOption = document.getElementById("irisOption");
    irisOption.style.display = (evaluation === "accuracy" || evaluation === "fm" || evaluation === "rand") ? "inline" : "none";

    var wineOption = document.getElementById("wineOption");
    wineOption.style.display = (evaluation === "auc" || evaluation === "f1" || evaluation === "pr" || evaluation === "roc") ? "none" : "inline";

}
function updatesplitter() {
    var percent = document.querySelector('input[name="percent"]:checked').value;

    var boostrapingOption = document.getElementById("boostrapingOption");
    boostrapingOption.style.display = (percent === "10%" || percent === "30%") ? "none" : "inline";

}
function resetForm() {
    var radioButtons = document.querySelectorAll('input[type="radio"]');

    for (var i = 0; i < radioButtons.length; i++) {
        radioButtons[i].checked = false;
    }
    location.reload(); // 刷新页面
}
