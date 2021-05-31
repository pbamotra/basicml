import { annotate } from 'https://unpkg.com/rough-notation@0.2.0/lib/rough-notation.esm.js?module';

const e = document.querySelector('#txnput');
const annotation = annotate(e, { type: 'underline', color: '#E86462'});
if (screen.availWidth >= 325) {
    annotation.show();
} else {
    annotation.hide();
}