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
    page.open("http://cn.mikecrm.com/zmABqIn");
    //page.open("sample.html");
  },
  function() {
    //Enter Credentials
    //setTimeout(function() {
      page.evaluate(function() {
        //document.querySelector("input[data-reactid='.0.0.0:$c201283423.1.0.$name.2']").value = "weiya";
        document.querySelector("input[data-reactid='.0.0.0:$c201283423.1.0.$name.2']").setAttribute('value', "weiya");
        document.querySelector("input[data-reactid='.0.0.0:$c201283425.1.0.2:$im0.0.2']").setAttribute('value', "17816889562");
        document.querySelector("a[id='form_submit']").click();	
        
        //document.querySelector('input[data-reactid=".0.0.0:$c201283425.1.0.2:$im0.0.2"]').value = "17816889562";
        //var d = document.querySelector("script[type='text/javascript']").text;
        //document.querySelector("script[type='text/javascript']").text = d.replace('{"lang":3};', '{"lang":3,"uvd":{"ct":{"23":{"n":"weiya"},"24":"17816889562"}}};');

        //console.log(document.querySelectorAll('html')[0].outerHTML);
      });  
    //}, 5000);
  }, 
  function() {
    //setTimeout(function() {
      page.evaluate(function() {
        var e = document.createEvent('MouseEvents');
        //e.initMouseEvent('click', true, true, window, 0, 0, 0, 0, 0, false, false, false, false, 0, null);
        e.initEvent("click", true, true);
        document.querySelector("a").dispatchEvent(e);
        //document.querySelector("a[id='form_submit']").click();	
        document.querySelector("a[id='form_submit']").dispatchEvent(e);	 
        //console.log(document.querySelectorAll('html')[0].outerHTML);
      });
    //}, 50000);
  }, 
  function() {
    // Output content of page to stdout after form has been submitted
    page.render('after_submit3.png')
    page.evaluate(function() {
	    console.log(document.querySelector("script[type='text/javascript']").text);
      //console.log(document.querySelectorAll('html')[0].outerHTML);
    });
  }
];


interval = setInterval(function() {
  if (!loadInProgress && typeof steps[testindex] == "function") {
    console.log("step " + (testindex + 1));
    page.render("step-" + (testindex + 1)+".png");
    //console.log(document.querySelectorAll('html')[0].outerHTML);
    steps[testindex]();
    testindex++;
  }
  if (typeof steps[testindex] != "function") {
    console.log("test complete!");
    phantom.exit();
  }
}, 1000);
