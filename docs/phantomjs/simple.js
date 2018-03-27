var page = require('webpage').create();
//var url = "http://cn.mikecrm.com/zmABqIn";
var url = "sample.html";

page.onConsoleMessage = function(msg) {
    console.log(msg);
  };
/*
setTimeout(function() {
    page.open(url);
}, 500000);

setTimeout(function() {
    console.log("fill form ...");
    page.evaluate(function() {
        console.log(document.querySelectorAll('html')[0].outerHTML);
//        document.querySelector("input[data-reactid='.0.0.0:$c201283423.1.0.$name.2']").value = "weiya";
//        document.querySelector('input[data-reactid=".0.0.0:$c201283425.1.0.2:$im0.0.2"]').value = "17816889562";
    });
}, 5000);

setTimeout(function() {
    console.log("show form ...");
    page.evaluate(function() {
        console.log(document.querySelectorAll('html')[0].outerHTML);
    });
}, 5000);


setTimeout(function() {
    setTimeout(function() {
        phantom.exit();
    }, 100);
}, 20000);*/

/*
page.open(url, function(){
    page.evaluate(function() {
        //console.log(document.querySelectorAll('html')[0].outerHTML);
        document.querySelector("input[data-reactid='.0.0.0:$c201283423.1.0.$name.2']").value = "weiya";
    },
    function(){
        setTimeout(function() {
            var html = page.evaluate(function(){
                return document.documentElement.outerHTML;
                //console.log(document.querySelectorAll('html')[0].outerHTML);
            });
            console.log(html);
        }, 5000)
    });
});

*/

page.open(url, function(){
    page.evaluate(function() {
        //console.log(document.querySelectorAll('html')[0].outerHTML);
        //console.log(document.querySelector("input[data-reactid='.0.0.0:$c201283423.1.0.$name.2']").className);
        document.querySelector("input[data-reactid='.0.0.0:$c201283423.1.0.$name.2']").value = "weiya";
        document.querySelector("input[data-reactid='.0.0.0:$c201283423.1.0.$name.2']").setAttribute('value', "weiya");
        document.querySelector('input[data-reactid=".0.0.0:$c201283425.1.0.2:$im0.0.2"]').value = "17816889562";
        document.querySelector("input[data-reactid='.0.0.0:$c201283425.1.0.2:$im0.0.2']").setAttribute('value', "17816889562");
        var d = document.querySelector("script[type='text/javascript']").text;
        document.querySelector("script[type='text/javascript']").text = d.replace('{"lang":3};', '{"lang":3,"uvd":{"ct":{"23":{"n":"weiya"},"24":"17816889562"}}};');
        //console.log(document.querySelector("input[data-reactid='.0.0.0:$c201283423.1.0.$name.2']").value);
    });
    setTimeout(function() { 
        page.evaluate(function(){
            //console.log(document.querySelector("input[data-reactid='.0.0.0:$c201283423.1.0.$name.2']").value);
            //return document.documentElement.outerHTML;
            console.log(document.querySelector('html').outerHTML);
            //console.log(document.querySelector("input[data-reactid='.0.0.0:$c201283423.1.0.$name.2']").value);
            document.querySelector("a[id='form_submit']").click();
            
        //    var e = document.createEvent('MouseEvents');
         //   e.initEvent("click", true, true);
           // document.querySelector("a[id='form_submit']").dispatchEvent(e);
        });
    }, 2000);
    setTimeout(function() {
        //page.render('simple.png');
        phantom.exit();
    }, 2000);
});