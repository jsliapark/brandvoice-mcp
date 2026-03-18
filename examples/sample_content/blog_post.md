# Building React Components That Scale

When I started building design systems at Netflix, the biggest lesson was this: every component should be a contract. Not just a visual element, but a promise about behavior, accessibility, and performance.

Here's what I mean. A Button component isn't just a styled `<button>` tag. It's a guarantee that keyboard navigation works, that focus states are visible, that loading states don't cause layout shift. When 200 engineers depend on your Button, you can't ship regressions.

The pattern that saved us? Compound components with context. Instead of prop drilling through 15 props, we composed behavior.

Three rules I follow now for every component:
1. Props are the public API. Type them ruthlessly.
2. Internal state should be invisible to consumers.
3. If you need more than 10 props, you need composition instead.

This isn't theoretical. These patterns ship at Netflix scale. Try them.
