export default function TermsPage() {
  return (
    <main className="mx-auto max-w-3xl px-6 py-12 text-[#151515]">
      <h1 className="text-3xl font-semibold">Terms of Service</h1>
      <p className="mt-4 text-sm leading-6 text-[#555555]">
        AdaptifAI - Creative Localization and Resizing Tool is provided by SASMAZ DIGITAL SOLUTIONS. AdaptifAI processes uploaded ad creatives only to provide OCR, translation, inpainting, resizing, preview and export services. Users are responsible for having the rights to upload, localize and publish the source creative assets.
      </p>
      <h2 className="mt-8 text-xl font-semibold">Credits and Payments</h2>
      <p className="mt-3 text-sm leading-6 text-[#555555]">
        Credit packs are billed through Stripe. Credits are consumed when an ADAPT job is successfully accepted by the processing pipeline.
      </p>
      <h2 className="mt-8 text-xl font-semibold">Temporary Files</h2>
      <p className="mt-3 text-sm leading-6 text-[#555555]">
        Creative uploads and generated files are temporary processing artifacts and are eligible for deletion after 24 hours.
      </p>
      <h2 className="mt-8 text-xl font-semibold">Business Contact</h2>
      <p className="mt-3 text-sm leading-6 text-[#555555]">
        İbrahim Tolgar ŞAŞMAZ / SASMAZ DIGITAL SOLUTIONS / 81543, Munich Germany / tolgar@sasmaz.digital
      </p>
    </main>
  );
}
