var page = require('webpage').create();
 
var system = require("system");
 
 
page.open("http://cn.mikecrm.com/zmABqIn", function (status) {
   
   
       /*
           page.onLoadFinished = function () {
           
            page.render("after_submit.png");
           
            phantom.exit();
           
        };
       */

        page.evaluate(function () {
           
	//	document.querySelector('input[name="id"]').value = "weiya";
		document.querySelector("input[data-reactid='.0.0.0:$c201283423.1.0.$name.2']").value = "weiya";
		//document.querySelector('form').submit();        
		document.querySelector('[data-reactid=".0.0.0:$c201283425.1.0.2:$im0.0.2"]').value = "17816889562";
		//document.querySelector("[data-reactid='.1.$submit.1']").click();
});

      
       
       window.setTimeout(function() {
            page.render("output.png");
            //page.close();
            console.log('finished...');
            //phantom.exit();
        }, 1000);
   var t = 10,
    interval = setInterval(function(){
        if ( t > 0 ) {
            console.log(t--);
        } else {
            console.log("BLAST OFF!");
            phantom.exit();
        }
    }, 1000);
           page.evaluate(function () {
           

		document.querySelector("[data-reactid='.1.$submit.1']").click();
});
   var t = 10,
    interval = setInterval(function(){
        if ( t > 0 ) {
            console.log(t--);
        } else {
            console.log("BLAST OFF!");
            phantom.exit();
        }
    }, 1000);

   window.setTimeout(function() {
//document.querySelector("[data-reactid='.1.$submit.1']").click();
            page.render("output2.png");
            //page.close();
            console.log('finished...');
            //phantom.exit();
        }, 1000);

   
});
