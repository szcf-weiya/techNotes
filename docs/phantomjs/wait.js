var page = require('webpage').create();

page.onResourceReceived = function(response) {
    if (response.stage !== "end") return;
    console.log('Response (#' + response.id + ', stage "' + response.stage + '"): ' + response.url);
};
page.onResourceRequested = function(requestData, networkRequest) {
    console.log('Request (#' + requestData.id + '): ' + requestData.url);
};
page.onUrlChanged = function(targetUrl) {
    console.log('New URL: ' + targetUrl);
};
page.onLoadFinished = function(status) {
    console.log('Load Finished: ' + status);
    page.render("test37_next_page.png");
};
page.onLoadStarted = function() {
    console.log('Load Started');
};
page.onNavigationRequested = function(url, type, willNavigate, main) {
    console.log('Trying to navigate to: ' + url);
};

page.open("http://cn.mikecrm.com/zmABqIn", function(status){
    page.evaluate(function() {
        document.querySelector("input[data-reactid='.0.0.0:$c201283423.1.0.$name.2']").value = "weiya";
        document.querySelector('input[data-reactid=".0.0.0:$c201283425.1.0.2:$im0.0.2"]').value = "17816889562";  
    });
    setTimeout(function() {
        page.evaluate(function() {
            var e = document.createEvent('MouseEvents');
            e.initMouseEvent('click', true, true, window, 0, 0, 0, 0, 0, false, false, false, false, 0, null);
            document.querySelector("a[id='form_submit']").dispatchEvent(e);	
        });    
    }, 10000);
    setTimeout(function() {}, 10000);
    setTimeout(function() {
        page.render("out.png");
        phantom.exit();
    }, 10000);
});