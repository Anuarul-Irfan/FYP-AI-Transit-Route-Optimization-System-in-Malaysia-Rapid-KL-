$(document).ready(function () {
    let segmentCount = 0;

    $('#preferences-header').click(function () {
        $('#preferences-form').slideToggle();
        $(this).find('i').toggleClass('fa-chevron-down fa-chevron-up');
    });
    // Handle slider value updates
    $('.slider-container input[type="range"]').on('input', function () {
        $(this).siblings('.slider-value').text($(this).val());
    });

    // Reset preferences button
    $('#reset-preferences').click(function () {
        $('#preferences-form input[type="range"]').val(0.5).trigger('input');
        $('#oku-friendly').prop('checked', false);
    });

    // Add destination button
    $('#add-destination').click(function () {
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

        // Enable the "Add Destination" button if this is the last segment
        if ($(`#to-station-${segmentCount}`).closest('.route-segment').is(':last-child')) {
            $('#add-destination').prop('disabled', false);
        }

        console.log('New segment added:', segmentCount); // Debugging
        console.log('New "from" station:', newFromStation.val()); // Debugging
    });

    // Remove segment button (delegated event handler)
    $('#routes-container').on('click', '.remove-segment', function () {
        const segment = $(this).closest('.route-segment');
        const segmentIndex = segment.data('segment');

        if (segmentIndex === 0) {
            return; // Don't remove the first segment
        }

        segment.remove();
        updateSegmentConnections();

        // Disable the "Add Destination" button if there are no more segments
        if ($('.route-segment').length === 1) {
            $('#add-destination').prop('disabled', true);
        }

        console.log('Segment removed:', segmentIndex); // Debugging
    });

    // Search routes button
    $('#search-routes').click(function () {
        const routes = [];
        let isValid = true;

        $('.route-segment').each(function () {
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

        console.log('Sending preferences:', preferences); // Debugging

        // Call API to find routes
        $.ajax({
            url: '/find_routes',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(preferences),
            success: function (routes) {
                console.log('Received routes:', routes); // Debugging
                displayRoutes(routes);
                $('.loading').hide();
                $('.results-panel').show(); // Ensure this line is executed
                $('.image-slider').hide(); // Hide slider when showing results
            },
            error: function (xhr) {
                $('.loading').hide();
                showError(xhr.responseJSON?.error || 'Error finding routes');
            }
        });
    });

    // Helper functions
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
        $('.route-segment').each(function (index) {
            if (index > 0) {
                $(this).before('<div class="route-connector"></div>');
            }
        });
    }

    function displayRoutes(routes) {
        const routeOptions = $('.route-options');
        routeOptions.empty(); // Clear existing routes

        if (routes.length === 0) {
            showError('No routes found');
            return;
        }

        // Display each route with its details
        routes.forEach((route, index) => {
            const routeCard = createRouteCard(route, index);
            const routeDetails = createRouteDetails(route, index);
            routeOptions.append(routeCard);
            routeOptions.append(routeDetails);
        });
    }
    function createRouteCard(route, index) {
        // Calculate total POIs
        const totalPois = route.steps.reduce((total, step) => {
            return total + (step.nearby_pois && Array.isArray(step.nearby_pois) ? step.nearby_pois.length : 0);
        }, 0);

        const eta = calculateETA(route.total_time);
        const template = `
        <div class="route-card" data-route-index="${index}">
            <div class="route-summary">
                <div class="route-header">
                    <h4>Route ${index + 1}</h4>
                    <div class="time-info">
                        <span class="route-duration">${route.total_time}</span>
                        <span class="route-eta">ETA: ${eta}</span>
                    </div>
                </div>
                <div class="route-info">
                    <div class="info-item">
                        <i class="fas fa-coins"></i>
                        <span>${route.total_cost}</span>
                    </div>
                    <div class="info-item">
                        <i class="fas fa-arrows-left-right"></i>
                        <span>${route.total_distance}</span>
                    </div>
                    <div class="info-item">
                        <i class="fas fa-exchange-alt"></i>
                        <span>${route.interchanges} transfers</span>
                    </div>
                    <div class="info-item">
                        <i class="fas fa-camera"></i>
                        <span>${totalPois} attractions</span>
                    </div>
                </div>
            </div>
            <button class="toggle-details-btn" data-target="#route-details-${index}">
                <i class="fas fa-chevron-down"></i>
            </button>
        </div>
    `;

        const card = $(template);
        card.find('.toggle-details-btn').click(function () {
            const target = $(this).data('target');
            $(target).slideToggle();
            $(this).find('i').toggleClass('fa-chevron-down fa-chevron-up');
        });

        return card;
    }
    function createRouteDetails(route, index) {
        console.log('Route data:', route); // Debug: Check entire route object
        console.log('Route steps:', route.steps); // Debug: Check steps array

        const totalPois = route.steps.reduce((total, step) => {
            console.log('Step POIs:', step.nearby_pois); // Debug: Check POIs for each step
            return total + (step.nearby_pois && Array.isArray(step.nearby_pois) ? step.nearby_pois.length : 0);
        }, 0);

        console.log('Total POIs:', totalPois); // Debug: Check total count

        const details = $(`<div class="route-details" id="route-details-${index}" style="display: none;"></div>`);

        // Add route summary with ETA and POI count
        details.append(`
        <div class="route-summary">
            <p><i class="fas fa-clock"></i> Total Duration: ${route.total_time}</p>
            <p><i class="fas fa-hourglass-end"></i> ETA: ${calculateETA(route.total_time)}</p>
            <p><i class="fas fa-coins"></i> Total Cost: ${route.total_cost}</p>
            <p><i class="fas fa-arrows-left-right"></i> Total Distance: ${route.total_distance}</p>
            <p><i class="fas fa-camera"></i> Total Points of Interest: ${totalPois}</p>
            <p><i class="fas fa-exchange-alt"></i> Total Transfers: ${route.interchanges}</p>
        </div>
    `);

        // Add step-by-step instructions with individual ETAs
        const steps = $('<div class="route-steps"></div>');
        let accumulatedMinutes = 0;

        route.steps.forEach((step, stepIndex) => {
            // Calculate ETA for this step
            const stepMinutes = parseInt(step.time_taken.split(' ')[0]);
            accumulatedMinutes += stepMinutes;
            const stepEta = calculateETA(`${accumulatedMinutes} mins`);

            // Add ETA to step data
            step.eta = stepEta;

            if (step.is_transfer || step.route_id === 'Walking') {
                // Walking/Transfer step
                steps.append(`
                <div class="step-container walk">
                    <div class="step-header">
                        <span class="line-badge" style="background-color: #64748b; color: white">
                            <i class="fas fa-walking"></i>
                            Walking
                        </span>
                        <div class="step-summary">
                            <p class="step-type">Transfer</p>
                            <p class="step-main">${step.from} → ${step.to}</p>
                        </div>
                    </div>
                    <div class="step-metrics">
                        <span><i class="fas fa-clock"></i> ${step.time_taken}</span>
                        <span><i class="fas fa-arrows-left-right"></i> ${step.distance}</span>
                        <span><i class="fas fa-coins"></i> ${step.cost}</span>
                    </div>
                    ${step.nearby_pois && step.nearby_pois.length > 0 ? `
                        <div class="step-pois">
                            <i class="fas fa-camera"></i>
                            <span>Points of Interest:</span>
                            <ul>
                                ${step.nearby_pois.map(poi => `<li>${poi}</li>`).join('')}
                            </ul>
                        </div>
                    ` : ''}
                </div>`);
            } else {
                // Transit step
                const lineColor = getLineColor(step.type, step.route_id);
                const fullLineName = getFullLineName(step.route_id);

                // Create POIs HTML if available
                const poisHtml = step.nearby_pois && step.nearby_pois.length > 0 ? `
                <div class="step-pois">
                    <i class="fas fa-camera"></i>
                    <span>Points of Interest:</span>
                    <ul>
                        ${step.nearby_pois.map(poi => `<li>${poi}</li>`).join('')}
                    </ul>
                </div>
            ` : '';

                steps.append(`
                <div class="step-container">
                    <div class="step-header transit">
                        <span class="line-badge" style="background-color: ${lineColor.bg}; color: ${lineColor.text}">
                            <i class="fas fa-${step.route_id === 'BRT' ? 'bus' : 'train'}"></i>
                            ${step.route_id}
                        </span>
                        <div class="step-summary">
                            <p class="step-type">${fullLineName}</p>
                            <p class="step-main">${step.from} → ${step.to}</p>
                            <p class="step-eta">ETA: ${step.eta}</p>
                        </div>
                    </div>
                    <div class="step-metrics">
                        <span><i class="fas fa-clock"></i> ${step.time_taken}</span>
                        <span><i class="fas fa-arrows-left-right"></i> ${step.distance}</span>
                        <span><i class="fas fa-coins"></i> ${step.cost}</span>
                    </div>
                    ${poisHtml}
                </div>`);
            }
        });

        details.append(steps);
        return details;
    }

    function calculateETA(totalTime) {
        const currentTime = new Date();
        const totalMinutes = parseInt(totalTime.split(' ')[0]);
        const eta = new Date(currentTime.getTime() + totalMinutes * 60000);
        return eta.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    function createStepHtml(step) {
        if (step.is_transfer || step.route_id === 'Walking') {
            // Walking/Transfer step
            return `
            <div class="step-container walk">
                <div class="step-header">
                    <span class="line-badge" style="background-color: #64748b; color: white">
                        <i class="fas fa-walking"></i>
                        Walking
                    </span>
                    <div class="step-summary">
                        <p class="step-type">Transfer</p>
                        <p class="step-main">${step.from} → ${step.to}</p>
                    </div>
                </div>
                <div class="step-metrics">
                    <span><i class="fas fa-clock"></i> ${step.time_taken}</span>
                    <span><i class="fas fa-arrows-left-right"></i> ${step.distance}</span>
                    <span><i class="fas fa-coins"></i> ${step.cost}</span>
                </div>
            </div>`;
        } else {
            // Transit step
            const lineColor = getLineColor(step.type, step.route_id);
            const fullLineName = getFullLineName(step.route_id);

            console.log('Step data:', step);
console.log('POIs:', step.nearby_pois);

            // Create POIs HTML if available
            const poisHtml = step.nearby_pois && step.nearby_pois.length > 0 ? `
            <div class="step-pois">
                <i class="fas fa-camera"></i>
                <span>Points of Interest:</span>
                <ul>
                    ${step.nearby_pois.map(poi => `<li>${poi}</li>`).join('')}
                </ul>
            </div>
        ` : '';

            return `
            <div class="step-container">
                <div class="step-header transit">
                    <span class="line-badge" style="background-color: ${lineColor.bg}; color: ${lineColor.text}">
                        <i class="fas fa-${step.route_id === 'BRT' ? 'bus' : 'train'}"></i>
                        ${step.route_id}
                    </span>
                    <div class="step-summary">
                        <p class="step-type">${fullLineName}</p>
                        <p class="step-main">${step.from} → ${step.to}</p>
                        <p class="step-eta">ETA: ${step.eta}</p>
                    </div>
                </div>
                <div class="step-metrics">
                    <span><i class="fas fa-clock"></i> ${step.time_taken}</span>
                    <span><i class="fas fa-arrows-left-right"></i> ${step.distance}</span>
                    <span><i class="fas fa-coins"></i> ${step.cost}</span>
                </div>
                ${poisHtml}
            </div>`;
        }
    }

    function getLineColor(type, route) {
        const routeToLine = {
            'KJL': 'KJ',
            'SPL': 'PH',
            'AGL': 'AG',
            'KGL': 'MRT',
            'PYL': 'PYL',
            'MRL': 'MR',
            'BRT': 'BRT'
        };

        const lineCode = routeToLine[route] || route;

        const lineColors = {
            'KJ': { bg: '#D50032', text: 'white' },
            'PH': { bg: '#76232F', text: 'white' },
            'AG': { bg: '#E57200', text: 'white' },
            'MRT': { bg: '#1A4731', text: 'white' },
            'PYL': { bg: '#FFCD00', text: 'black' },
            'MR': { bg: '#84BD00', text: 'white' },
            'BRT': { bg: '#115740', text: 'white' }
        };
        return lineColors[lineCode] || { bg: '#666', text: 'white' };
    }

    function getFullLineName(lineCode) {
        const lineNames = {
            'AGL': 'Ampang Line',
            'KJL': 'Kelana Jaya Line',
            'SPL': 'Sri Petaling Line',
            'KGL': 'Kajang Line',
            'PYL': 'Putrajaya Line',
            'MRL': 'Monorail Line',
            'BRT': 'BRT Sunway Line'
        };
        return lineNames[lineCode] || lineCode; // Return the full name or the code if not found
    }

    function showError(message) {
        $('.error-message').text(message).show();
    }

    // Initialize autocomplete for the first segment
    initializeAutocomplete($('#from-station-0'));
    initializeAutocomplete($('#to-station-0'));

    // Disable the "Add Destination" button initially
    $('#add-destination').prop('disabled', true);
});