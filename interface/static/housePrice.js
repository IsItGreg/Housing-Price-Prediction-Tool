
function calculatePrice(){
    fillDefaults();

    let inputs = document.getElementsByClassName("inpel"), valueList = {};
    console.log(inputs.length);
    for(let i = 0; i < inputs.length; ++i){
        valueList[inputs[i].id] = inputs[i].value;
    }

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

    if(document.getElementById("inpnumfloor").value === "")
        document.getElementById("inpnumfloor").value = 1;

    if(document.getElementById("inpnumpark").value === "")
        document.getElementById("inpnumpark").value = 1;

    if(document.getElementById("inptotalrooms").value === "")
        document.getElementById("inptotalrooms").value = 3;

    if(document.getElementById("inpbedrooms").value === "")
        document.getElementById("inpbedrooms").value = 1;

    if(document.getElementById("inpnumfbrooms").value === "")
        document.getElementById("inpnumfbrooms").value = 1;

    if(document.getElementById("inpnumhbrooms").value === "")
        document.getElementById("inpnumhbrooms").value = 0;


}
