var buttonStates = {
  accuracyOption: false,
  aucOption: false,
  distanceOption:false,
  f1Option: false,
  fmOption:false,
  mseOption:false,
  prOption: false,
  randOption:false,
  rmseOption:false,
  rocOption: false

};

function updateOptions() {
    var dataset = document.querySelector('input[name="Dataset"]:checked').value;

    buttonStates.accuracyOption = true;
    buttonStates.fmOption = true;
    buttonStates.randOption = true;
    buttonStates.distanceOption = dataset !== "iris";
    buttonStates.mseOption = dataset !== "iris";
    buttonStates.rmseOption = dataset !== "iris";
    buttonStates.f1Option = dataset === "heart";
    buttonStates.prOption = dataset === "heart";
    buttonStates.aucOption = dataset === "heart";
    buttonStates.rocOption = dataset === "heart";

//    updateevaluationbymodel(dataset)


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
    percentOption.style.display = (splitter === "holdout") ? "block" : "none";
}

function updatesplitter() {
    var percent = document.querySelector('input[name="percent"]:checked').value;

    var boostrapingOption = document.getElementById("boostrapingOption");
    boostrapingOption.style.display = (percent === "10%" || percent === "30%") ? "none" : "inline";

}



function updateevaluationbymodel() {
    var model = document.querySelector('input[name="model"]:checked').value;

    var accuracyOption = document.getElementById("accuracyOption");
    accuracyOption.style.display = (buttonStates.accuracyOption && (model === "GradientBoosting" || model === "KNN" ||
    model === "NB" || model === "Random Forest" || model === "LogisticRegression" || model === "SVM")) ? "inline" : "none";

    var distanceOption = document.getElementById("distanceOption");
    distanceOption.style.display = (buttonStates.distanceOption && model === "K_means") ? "inline" : "none";

    var f1Option = document.getElementById("f1Option");
    f1Option.style.display = (buttonStates.f1Option && (model === "GradientBoosting" || model === "KNN" ||
    model === "NB" || model === "Random Forest" || model === "LogisticRegression" || model === "SVM")) ? "inline" : "none";

    var fmOption = document.getElementById("fmOption");
    fmOption.style.display = (buttonStates.fmOption && model === "K_means") ? "inline" : "none";

    var mseOption = document.getElementById("mseOption");
    mseOption.style.display = (buttonStates.mseOption && (model === "Decision Tree" || model === "KNN" ||
    model === "LR" || model === "Random Forest" || model === "LogisticRegression" || model === "SVM")) ? "inline" : "none";

    var prOption = document.getElementById("prOption");
    prOption.style.display = (buttonStates.prOption && (model === "GradientBoosting" || model === "KNN" ||
    model === "LogisticRegression")) ? "inline" : "none";

    var randOption = document.getElementById("randOption");
    randOption.style.display = (buttonStates.randOption && model === "K_means") ? "inline" : "none";

    var aucOption = document.getElementById("aucOption");
    aucOption.style.display = (buttonStates.aucOption && (model === "GradientBoosting" || model === "KNN" ||
    model === "Random Forest" || model === "LogisticRegression")) ? "inline" : "none";

    var rmseOption = document.getElementById("rmseOption");
    rmseOption.style.display = (buttonStates.rmseOption && (model === "Decision Tree" || model === "KNN" ||
    model === "LR" || model === "Random Forest" || model === "LogisticRegression" || model === "SVM")) ? "inline" : "none";

    var rocOption = document.getElementById("rocOption");
    rocOption.style.display = (buttonStates.rocOption && (model === "GradientBoosting" || model === "KNN" ||
    model === "Random Forest" || model === "LogisticRegression" || model === "SVM")) ? "inline" : "none";


}

function resetForm() {
    var radioButtons = document.querySelectorAll('input[type="radio"]');

    for (var i = 0; i < radioButtons.length; i++) {
        radioButtons[i].checked = false;
    }
    location.reload(); // 刷新页面
}
