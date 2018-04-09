myUrl = 'http://www.google.com'

var phantom = Meteor.npmRequire('phantom')
phantom.create = Meteor.wrapAsync(phantom.create)
phantom.create( function(ph) {

    ph.createPage = Meteor.wrapAsync(ph.createPage)
    ph.createPage(function(page) {

        page.open = Meteor.wrapAsync(page.open)
        page.open(listingUrl, function(status) {
            console.log('Page loaded')

            page.evaluate = Meteor.wrapAsync(page.evaluate)
            page.evaluate(function() {

                // Find the button
                var element = document.querySelector( '.search-btn' );

                // create a mouse click event
                var event = document.createEvent( 'MouseEvents' );
                event.initMouseEvent( 'click', true, true, window, 1, 0, 0 );

                // send click to element
                element.dispatchEvent( event );

                // Give page time to process Click event
                setTimeout(function() {
                    // Return HTML code
                    return document.documentElement.outerHTML
                }, 5000)

            }, function(html) {

                // html is `null`
                doSomething()

            })
        })
    })
})
