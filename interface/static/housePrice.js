
function calculatePrice(){
    fillDefaults();

    // document.getElementById("housePrice").value =
    //     document.getElementById("inpsqft").value * 0.1 *
    //     document.getElementById("inpyrblt").value * 0.03 *
    //     document.getElementById("inpyrrem").value * 0.02;
    let inputs = document.getElementsByClassName("inpel"), valueList = {};
    console.log(inputs.length);
    for(let i = 0; i < inputs.length; ++i){
        valueList[inputs[i].id] = inputs[i].value;
    }
    let retData = '';
    $.getJSON('receiver', valueList, function(data){
        document.getElementById("housePrice").value = data.result;
    });

}

function fillDefaults(){
    let inputs = document.getElementsByClassName("inpel");
    for(let i = 0; i < inputs.length; ++i){
        if(inputs[i].value === ""){
            inputs[i].style.backgroundColor = "palegreen";
            inputs[i].setAttribute("onchange", 'this.style.backgroundColor = ""')
        }
    }

    if(document.getElementById("inpsqft").value === "")
        document.getElementById("inpsqft").value = 1000;

    if(document.getElementById("inpyrblt").value === "")
        document.getElementById("inpyrblt").value = 1980;

    if(document.getElementById("inpyrrem").value === "")
        document.getElementById("inpyrrem").value = 2000;
}
