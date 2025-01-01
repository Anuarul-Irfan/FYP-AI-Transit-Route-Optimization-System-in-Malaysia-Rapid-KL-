$(document).ready(function() {
    let segmentCount = 0;

    // Handle slider value updates
    $('.slider-container input[type="range"]').on('input', function() {
        $(this).siblings('.slider-value').text($(this).val());
    });

    // Reset preferences button
    $('#reset-preferences').click(function() {
        $('#preferences-form input[type="range"]').val(0.5).trigger('input');
        $('#oku-friendly').prop('checked', false);
    });

    // Add destination button
    $('#add-destination').click(function() {
        const lastSegment = $('.route-segment').last();
        const lastToStation = lastSegment.find('.to-station');
        const lastToStationInfo = lastSegment.find('.to-station').closest('.input-with-icon').find('.selected-station-info');

        if (!lastToStation.data('station-id')) {
            showError('Please select a destination before adding a new one');
            return;
        }

        segmentCount++;
        const newSegment = createRouteSegment(segmentCount);
        $('#routes-container').append(newSegment);

        // Set the "from" station of the new segment to the "to" station of the last segment
        const newFromStation = $(`#from-station-${segmentCount}`);
        newFromStation.val(lastToStation.val());
        newFromStation.data('station-id', lastToStation.data('station-id'));
        newFromStation.prop('readonly', true);

        // Copy the station info display
        const newFromStationInfo = newFromStation.closest('.input-with-icon').find('.selected-station-info');
        newFromStationInfo.html(lastToStationInfo.html());

        // Initialize autocomplete for the new "to" station input
        initializeAutocomplete($(`#to-station-${segmentCount}`));
    });

    // Remove segment button (delegated event handler)
    $('#routes-container').on('click', '.remove-segment', function() {
        const segment = $(this).closest('.route-segment');
        const segmentIndex = segment.data('segment');
        
        if (segmentIndex === 0) {
            return; // Don't remove the first segment
        }

        segment.remove();
        updateSegmentConnections();
    });

    // Search routes button
    $('#search-routes').click(function() {
        const routes = [];
        let isValid = true;

        $('.route-segment').each(function() {
            const fromStation = $(this).find('.from-station').data('station-id');
            const toStation = $(this).find('.to-station').data('station-id');

            if (!fromStation || !toStation) {
                isValid = false;
                return false; // break the loop
            }

            routes.push({
                from: fromStation,
                to: toStation
            });
        });

        if (!isValid) {
            showError('Please select all stations before searching');
            return;
        }

        // Show loading state
        $('.loading').show();
        $('.results-panel, .error-message').hide();

        // Collect preferences
        const preferences = {
            cost: parseFloat($('#cost-priority').val()),
            comfort: parseFloat($('#comfort-priority').val()),
            scenic: parseFloat($('#scenic-priority').val()),
            distance: parseFloat($('#distance-priority').val()),
            oku: $('#oku-friendly').is(':checked'),
            routes: routes
        };

        // Call API to find routes
        $.ajax({
            url: '/find_routes',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(preferences),
            success: function(routes) {
                displayRoutes(routes);
                $('.loading').hide();
                $('.results-panel').show();
            },
            error: function(xhr) {
                $('.loading').hide();
                showError(xhr.responseJSON?.error || 'Error finding routes');
            }
        });
    });

    function createRouteSegment(index) {
        return $(`
            <div class="route-segment" data-segment="${index}">
                ${index > 0 ? '<button class="remove-segment"><i class="fas fa-times"></i></button>' : ''}
                <div class="station-input">
                    <label for="from-station-${index}">From:</label>
                    <div class="input-with-icon">
                        <input type="text" id="from-station-${index}" class="from-station" placeholder="Enter starting station..." ${index > 0 ? 'readonly' : ''}>
                        <div class="selected-station-info"></div>
                    </div>
                </div>

                <div class="station-input">
                    <label for="to-station-${index}">To:</label>
                    <div class="input-with-icon">
                        <input type="text" id="to-station-${index}" class="to-station" placeholder="Enter destination station...">
                        <div class="selected-station-info"></div>
                    </div>
                </div>
            </div>
        `);
    }

    function updateSegmentConnections() {
        $('.route-connector').remove();
        $('.route-segment').each(function(index) {
            if (index > 0) {
                $(this).before('<div class="route-connector"></div>');
            }
        });
    }

    function displayRoutes(routes) {
        const routeOptions = $('.route-options');
        routeOptions.empty();

        routes.forEach((route, index) => {
            const routeCard = createRouteCard(route, index);
            routeOptions.append(routeCard);
        });

        if (routes.length > 0) {
            showRouteDetails(routes[0], 0);
        }
    }

    function createRouteCard(route, index) {
        const template = `
            <div class="route-card ${index === 0 ? 'selected' : ''}" data-route-index="${index}">
                <div class="route-summary">
                    <div class="route-header">
                        <h4>Route ${index + 1}</h4>
                        <span class="route-duration">${route.duration}</span>
                    </div>
                    <div class="route-info">
                        <div class="info-item">
                            <i class="fas fa-coins"></i>
                            <span>${route.cost}</span>
                        </div>
                        <div class="info-item">
                            <i class="fas fa-arrows-left-right"></i>
                            <span>${route.distance}</span>
                        </div>
                        <div class="info-item">
                            <i class="fas fa-exchange-alt"></i>
                            <span>${route.interchanges} transfers</span>
                        </div>
                    </div>
                </div>
            </div>
        `;

        const card = $(template);
        card.click(function() {
            $('.route-card').removeClass('selected');
            $(this).addClass('selected');
            showRouteDetails(route, index);
        });

        return card;
    }

    function showRouteDetails(route, index) {
        const detailsPanel = $('.route-details');
        detailsPanel.empty();

        // Add route header
        detailsPanel.append(`
            <h3>Route ${index + 1} Details</h3>
            <div class="route-summary">
                <p><i class="fas fa-clock"></i> ${route.duration}</p>
                <p><i class="fas fa-coins"></i> ${route.cost}</p>
                <p><i class="fas fa-arrows-left-right"></i> ${route.distance}</p>
                <p><i class="fas fa-exchange-alt"></i> ${route.interchanges} transfers</p>
            </div>
        `);

        // Add scenic spots if available
        if (route.scenic_spots && route.scenic_spots.length > 0) {
            detailsPanel.append(`
                <div class="scenic-spots">
                    <h4>Nearby Attractions:</h4>
                    <ul>
                        ${route.scenic_spots.map(spot => `<li>${spot}</li>`).join('')}
                    </ul>
                </div>
            `);
        }

        // Add step-by-step instructions
        const steps = $('<div class="route-steps"></div>');
        route.steps.forEach(step => {
            const stepHtml = createStepHtml(step);
            steps.append(stepHtml);
        });

        detailsPanel.append(steps);
    }

    function createStepHtml(step) {
        if (step.type === 'walk') {
            return `
                <div class="step walk">
                    <i class="fas fa-walking"></i>
                    <div class="step-details">
                        <p class="step-type">Walking transfer</p>
                        <p>${step.from} → ${step.to}</p>
                        <p class="step-duration">${step.duration}</p>
                    </div>
                </div>`;
        } else {
            const lineColor = getLineColor(step.type, step.line);
            return `
                <div class="step transit">
                    <span class="line-badge" style="background-color: ${lineColor.bg}; color: ${lineColor.text}">
                        <i class="fas fa-${step.type === 'BRT' ? 'bus' : 'train'}"></i>
                        ${step.line}
                    </span>
                    <div class="step-details">
                        <p class="step-type">${step.type}</p>
                        <p>${step.from} → ${step.to}</p>
                    </div>
                </div>`;
        }
    }

    function getLineColor(type, route) {
        const routeToLine = {
            'KJL': 'KJ',
            'SPL': 'PH',
            'AGL': 'AG',
            'KJG': 'MRT',
            'PYL': 'PYL',
            'MRL': 'MR',
            'BRT': 'BRT'
        };
        
        const lineCode = routeToLine[route] || route;
        
        const lineColors = {
            'KJ': { bg: '#D50032', text: 'white' },     // Red
            'PH': { bg: '#76232F', text: 'white' },     // Brown
            'AG': { bg: '#E57200', text: 'white' },     // Orange
            'MRT': { bg: '#1A4731', text: 'white' },    // Dark Green
            'PYL': { bg: '#FFCD00', text: 'black' },    // Yellow
            'MR': { bg: '#84BD00', text: 'white' },     // Light Green
            'BRT': { bg: '#115740', text: 'white' }     // Darkest Green
        };
        return lineColors[lineCode] || { bg: '#666', text: 'white' };
    }

    function showError(message) {
        $('.error-message').text(message).show();
    }

    // Initialize autocomplete for the first segment
    initializeAutocomplete($('#from-station-0'));
    initializeAutocomplete($('#to-station-0'));
}); 