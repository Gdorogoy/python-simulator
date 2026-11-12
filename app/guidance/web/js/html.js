// `html` lets us write markup inside JavaScript without a build step (it is the "htm" library bound to React):
//
//     html`<button onClick=${save}>Save</button>`           // an HTML element
//     html`<${Card} title="Scenario">...</${Card}>`          // one of our own components
//
// Inside the template: ${value} inserts a value, className / htmlFor are used instead of class / for.
import React from 'react';
import htm from 'htm';

export const html = htm.bind(React.createElement);
