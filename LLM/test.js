// Initialize the ECharts instance
var myChart = echarts.init(document.getElementById('main'));

// Fetch the JSON data
fetch('sushi_echart.json')
    .then(response => response.json())
    .then(data => {
        // Prepare the series data for ECharts
        const nodes = data.nodes.map(node => ({
            name: node.id,
            value: node.id
        }));
        
        const links = data.links.map(link => ({
            source: link.source,
            target: link.target,
            name: link.relationship
        }));

        // ECharts option
        var option = {
            title: {
                text: 'Graph Visualization',
                subtext: 'Using ECharts.js',
                left: 'center'
            },
            tooltip: {},
            animation: false,
            series: [
                {
                    type: 'graph',
                    layout: 'force', // Use force layout for better visualization
                    data: nodes,
                    links: links,
                    roam: true, // Allow zooming and panning
                    label: {
                        show: true,
                        fontSize: 11
                    },
                    edgeLabel: {
                        show: true,
                        formatter: function (params) {
                            return params.data.name; // Display the relationship name
                        },
                        fontSize: 9
                    },
                    lineStyle: {
                        color: '#aaa'
                    },
                    emphasis: {
                        focus: 'adjacency'
                    },
                    symbolSize: 30, 
                    // Additional settings
                    force: {
                        repulsion: 500 // Adjust repulsion for node spacing
                    }
                }
            ]
        };

        // Set the option and render the chart
        myChart.setOption(option);
    })
    .catch(error => console.error('Error loading the JSON data:', error));
