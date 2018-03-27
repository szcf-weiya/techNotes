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
    page.open("toy.html");
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
	//	document.querySelector("input[data-reactid='.0.0.0:$c201283423.1.0.$name.2']").value = "weiya";
	document.querySelector('input[name="id"]').value = "weiya";
	document.querySelector('form').submit();
    });
      
  }, 
  function() {
    //Login
    page.evaluate(function() {
//      var arr = document.getElementsByClassName("form");
//      var i;

//      for (i=0; i < arr.length; i++) {
//        if (arr[i].getAttribute('method') == "POST") {
//        ar.submit();
//          return;
//        }
//      }
//	document.querySelector("[type='submit']").click();
    });
  }, 
  function() {
    // Output content of page to stdout after form has been submitted
    page.evaluate(function() {
      console.log(document.querySelectorAll('html')[0].outerHTML);
    });
  }
];


interval = setInterval(function() {
  if (!loadInProgress && typeof steps[testindex] == "function") {
    console.log("step " + (testindex + 1));
    steps[testindex]();
    testindex++;
  }
  if (typeof steps[testindex] != "function") {
    console.log("test complete!");
    phantom.exit();
  }
}, 50);
