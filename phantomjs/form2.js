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
  },
  function() {
    //Enter Credentials
   	        page.evaluate(function () {
           
	//	document.querySelector('input[name="id"]').value = "weiya";
		document.querySelector("input[data-reactid='.0.0.0:$c201283423.1.0.$name.2']").value = "weiya";
		//document.querySelector('form').submit();        
		document.querySelector('[data-reactid=".0.0.0:$c201283425.1.0.2:$im0.0.2"]').value = "17816889562";
		document.querySelector("[data-reactid='.1.$submit.1']").click();
});
    });
  }, 
  function() {
    //Login
   page.render("before_submit1.png");
  }, 
  function() {
    // Output content of page to stdout after form has been submitted
      page.render("after_submit2.png");
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
}, 5000);
