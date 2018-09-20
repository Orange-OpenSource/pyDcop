from pydcop.commands.generators.meetingscheduling import (
    generate_resources,
    generate_events,
    generate_problem_definition,
    peav_variables_for_resource,
)


def test_generate_resources():
    resources = generate_resources(count=3, max_value=5, slots=[1, 2, 3])

    assert len(resources) == 3
    for id, resource in resources.items():
        assert id == resource.id
        assert len(resource.value_free) == 3
        for value in resource.value_free:
            assert 0 <= value <= 5


def test_generate_events():
    resources = generate_resources(count=3, max_value=5, slots=[1, 2, 3])
    events = generate_events(
        count=20,
        max_value=5,
        max_length=2,
        resources=list(resources.values()),
        max_resources_count=3,
    )

    assert len(events) == 20
    for id, event in events.items():
        assert event.id == id
        assert 1 <= event.length <= 5
        assert 1 <= len(event.resources) <= 3
        assert len(set(event.resources)) == len(event.resources)
        for resource, value in event.resources.items():
            assert 0 <= value <= 5


def test_generate_variables():
    slots_count = 10
    slots, events, resources = generate_problem_definition(
        slots_count=slots_count,
        resources_count=5,
        max_resource_value=10,
        events_count=3,
        max_length_event=2,
        max_resources_event=3,
    )

    _, resource = resources.popitem()
    variables = peav_variables_for_resource(resource, events, slots_count)
    events_with_resource = [
        evt for evt in events.values() if resource.id in evt.resources
    ]
    assert len(variables) == len(events_with_resource)
