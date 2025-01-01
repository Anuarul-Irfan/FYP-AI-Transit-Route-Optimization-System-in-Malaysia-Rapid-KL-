$(document).ready(function() {
    window.initializeAutocomplete = function(input) {
        input.autocomplete({
            source: function(request, response) {
                $.get('/api/stations', { term: request.term }, function(data) {
                    response(data.filter(station => 
                        station.label.toLowerCase().includes(request.term.toLowerCase())
                    ));
                });
            },
            minLength: 2,
            select: function(event, ui) {
                $(this).val(ui.item.label);
                $(this).data('station-id', ui.item.value);
                updateSelectedStation($(this), ui.item);
                return false;
            }
        }).each(function() {
            $(this).data('ui-autocomplete')._renderItem = function(ul, item) {
                return $('<li>')
                    .append(formatStation(item))
                    .appendTo(ul);
            };
        });

        // Clear station ID and info when input is cleared
        input.on('input', function() {
            if (!$(this).val()) {
                $(this).removeData('station-id');
                $(this).closest('.input-with-icon').find('.selected-station-info').empty();
            }
        });
    };

    function getLineColor(type, route) {
        // Convert route ID to line code if needed
        const routeToLine = {
            'KJL': 'KJ',
            'SPL': 'PH',
            'AGL': 'AG',
            'KJG': 'MRT',  // Kajang Line
            'PYL': 'PYL',
            'MRL': 'MR',
            'BRT': 'BRT'
        };
        
        const lineCode = routeToLine[route] || route;
        
        const lineColors = {
            'KJ': { bg: '#D50032', text: 'white' },     // Red - Kelana Jaya Line
            'PH': { bg: '#76232F', text: 'white' },     // Brown - Sri Petaling Line
            'AG': { bg: '#E57200', text: 'white' },     // Orange - Ampang Line
            'MRT': { bg: '#1A4731', text: 'white' },    // Dark Green - Kajang Line
            'PYL': { bg: '#FFCD00', text: 'black' },    // Yellow - Putrajaya Line
            'MR': { bg: '#84BD00', text: 'white' },     // Light Green - Monorail
            'BRT': { bg: '#115740', text: 'white' }     // Darkest Green - BRT Sunway
        };
        return lineColors[lineCode] || { bg: '#666', text: 'white' };
    }

    function formatStation(station) {
        const lineColor = getLineColor(station.type, station.route);
        return $(`<div class="station-item">
            <span class="line-badge" style="background-color: ${lineColor.bg}; color: ${lineColor.text}">
                <i class="fas fa-${station.type === 'BRT' ? 'bus' : 'train'}" style="margin-right: 4px;"></i>
                ${station.route}
            </span>
            <span class="station-name">${station.label}</span>
            ${station.isOKU ? '<i class="fas fa-wheelchair"></i>' : ''}
        </div>`);
    }

    function updateSelectedStation(input, station) {
        const container = input.closest('.input-with-icon').find('.selected-station-info');
        const lineColor = getLineColor(station.type, station.route);
        
        container.html(`
            <span class="line-badge" style="background-color: ${lineColor.bg}; color: ${lineColor.text}">
                <i class="fas fa-${station.type === 'BRT' ? 'bus' : 'train'}" style="margin-right: 4px;"></i>
                ${station.route}
            </span>
            <span class="station-name">${station.label}</span>
            ${station.isOKU ? '<i class="fas fa-wheelchair"></i>' : ''}
        `);

        // If this is a "to" station and it's the last segment, enable the "Add Destination" button
        if (input.hasClass('to-station') && input.closest('.route-segment').is(':last-child')) {
            $('#add-destination').prop('disabled', false);
        }
    }

    // Initialize autocomplete for the initial inputs
    initializeAutocomplete($('#from-station-0'));
    initializeAutocomplete($('#to-station-0'));

    // Disable the "Add Destination" button initially
    $('#add-destination').prop('disabled', true);
}); 