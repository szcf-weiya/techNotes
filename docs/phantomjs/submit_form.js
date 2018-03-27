var page = new WebPage(), testindex = 0, loadInProgress = false;

page.onConsoleMessage = function(msg) {
  console.log(msg);
};

page.onLoadStarted = function() {
  loadInProgress = true;
  console.log("load started");
};

page.onLoadFinished = function() {
  loadInProgress = false;
  console.log("load finished");
};

var steps = [
  function() {
    //Load Login Page
    page.open("http://cn.mikecrm.com/zmABqIn");
//page.open("sample.html");
  },
  function() {
    //Enter Credentials
    page.evaluate(function() {
/*
      var arr = document.getElementsByClassName("form");
      var i;

      for (i=0; i < arr.length; i++) { 

	  arr[i].elements["name"].value="mylogin";
          arr[i].elements["im"].value="mypassword";
          return;
      }
*/
//var d = document.querySelector("script[type='text/javascript']").text;
//document.querySelector("script[type='text/javascript']").text = d.replace('{"lang":3};', '{"lang":3,"uvd":{"ct":{"23":{"n":"weiya"},"24":"17816889562"}}};');

//console.log(document.querySelector("script[type='text/javascript']").text);
		document.querySelector("input[data-reactid='.0.0.0:$c201283423.1.0.$name.2']").value = "weiya";
		//document.querySelector('[aria-labelledby="title201283423 subtitle201283423 placehoder201283423"]').value = "weiya";
	document.querySelector('[data-reactid=".0.0.0:$c201283425.1.0.2:$im0.0.2"]').value = "17816889562";
	      //console.log(document.querySelectorAll('html')[0].outerHTML);
    });
  }, 
  function() {
    //Login
    page.evaluate(function() {
       window.setTimeout(function() {
	document.querySelector("a[id='form_submit']").click();
            page.render("output.png");
            //page.close();
            console.log('finished...');
            //phantom.exit();
        }, 1000);
/*
      var arr = document.getElementsByTagName("form");
//arr.submit();
      var i;
      for (i=0; i < arr.length; i++) {
        if (arr[i].getAttribute('method') == "POST") {
          arr[i].submit();
          return;
        }
      }*/
//	document.querySelector("a[id='form_submit']").click();
//console.log(document.querySelector("script[type='text/javascript']").text);
//   console.log(document.querySelectorAll('html')[0].outerHTML);
 
    });
  }, 
  function() {
    // Output content of page to stdout after form has been submitted
    page.evaluate(function() {
	console.log(document.querySelector("script[type='text/javascript']").text);
  //    console.log(document.querySelectorAll('html')[0].outerHTML);
    });
  }
];


interval = setInterval(function() {
  if (!loadInProgress && typeof steps[testindex] == "function") {
    console.log("step " + (testindex + 1));
    page.render("step-" + (testindex + 1)+".png");

    steps[testindex]();
    testindex++;
  }
  if (typeof steps[testindex] != "function") {
    console.log("test complete!");
    phantom.exit();
  }
}, 100);
