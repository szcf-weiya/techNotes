var FindProxyForURL = function(init, profiles) {
    return function(url, host) {
        "use strict";
        var result = init, scheme = url.substr(0, url.indexOf(":"));
        do {
            result = profiles[result];
            if (typeof result === "function") result = result(url, host, scheme);
        } while (typeof result !== "string" || result.charCodeAt(0) === 43);
        return result;
    };
}("+gmail", {
    "+gmail": function(url, host, scheme) {
        "use strict";
        if (/(?:^|\.)gmail\.com$/.test(host)) return "+CUHK";
        if (/(?:^|\.)google\.com$/.test(host)) return "+CUHK";
        if (/(?:^|\.)googleusercontent\.com$/.test(host)) return "+CUHK";
        if (/(?:^|\.)gstatic\.com$/.test(host)) return "+CUHK";
        if (/(?:^|\.)googleapis\.com$/.test(host)) return "+CUHK";
        return "DIRECT";
    },
    "+CUHK": function(url, host, scheme) {
        "use strict";
        if (/^127\.0\.0\.1$/.test(host) || /^::1$/.test(host) || /^localhost$/.test(host)) return "DIRECT";
        return "SOCKS5 127.0.0.1:30002; SOCKS 127.0.0.1:30002";
    }
});