'use client';

import Link from 'next/link';
import brandData from '@/data/brand-social-data.json';

export default function BrandsPage() {
  return (
    <main style={{ padding: '40px', maxWidth: '1200px', margin: '0 auto', fontFamily: 'var(--font-sans, system-ui)' }}>
      <header style={{ marginBottom: '40px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ width: '32px', height: '32px', border: '2px solid var(--accent-primary, #0070f3)', borderRadius: '8px' }}></div>
          <h1 style={{ fontSize: '1.5rem', fontWeight: 600, margin: 0 }}>
            Floor<span style={{ color: 'var(--accent-primary, #0070f3)' }}>2</span>Feed
          </h1>
        </div>
        <Link href="/" style={{ color: 'var(--text-secondary, #666)', textDecoration: 'none', fontWeight: 500 }}>
          &larr; Back to Home
        </Link>
      </header>

      <div style={{ marginBottom: '32px' }}>
        <h2 style={{ fontSize: '2rem', fontWeight: 700, marginBottom: '8px', color: 'var(--text-primary, #111)' }}>
          Brand Social Monitor
        </h2>
        <p style={{ color: 'var(--text-secondary, #666)', fontSize: '1.1rem' }}>
          Tracking Instagram activity and social links for {brandData.length} brands.
        </p>
      </div>

      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', 
        gap: '24px' 
      }}>
        {brandData.map((brand: any, idx: number) => (
          <div key={idx} style={{
            background: 'var(--bg-card, #fff)',
            border: '1px solid var(--border-color, #eaeaea)',
            borderRadius: '16px',
            padding: '24px',
            boxShadow: '0 4px 14px rgba(0,0,0,0.05)',
            transition: 'transform 0.2s ease, box-shadow 0.2s ease',
            cursor: 'default'
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = 'translateY(-4px)';
            e.currentTarget.style.boxShadow = '0 8px 24px rgba(0,0,0,0.1)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = 'translateY(0)';
            e.currentTarget.style.boxShadow = '0 4px 14px rgba(0,0,0,0.05)';
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
              <h3 style={{ fontSize: '1.25rem', fontWeight: 600, margin: 0 }}>{brand.name}</h3>
              {brand.instagramActivity && (
                <span style={{
                  background: 'rgba(225, 48, 108, 0.1)',
                  color: '#E1306C',
                  padding: '4px 8px',
                  borderRadius: '20px',
                  fontSize: '0.75rem',
                  fontWeight: 600
                }}>Active</span>
              )}
            </div>

            <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', marginBottom: '20px' }}>
              {Object.entries(brand.links).map(([platform, url]) => (
                <a key={platform} href={url as string} target="_blank" rel="noreferrer" style={{
                  fontSize: '0.875rem',
                  color: 'var(--accent-primary, #0070f3)',
                  textDecoration: 'none',
                  textTransform: 'capitalize',
                  background: 'var(--bg-wash, #f5f5f5)',
                  padding: '4px 10px',
                  borderRadius: '8px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px'
                }}>
                  {platform.replace('_', ' ')}
                </a>
              ))}
            </div>

            {brand.instagramActivity && (
              <div style={{ marginBottom: '16px', fontSize: '0.875rem', color: 'var(--text-secondary, #666)' }}>
                <strong>Stats:</strong> {brand.instagramActivity}
              </div>
            )}

            {brand.lastPosts && brand.lastPosts.length > 0 && (
              <div>
                <h4 style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: '8px', color: 'var(--text-primary, #111)', borderBottom: '1px solid var(--border-color, #eaeaea)', paddingBottom: '4px' }}>Recent Posts</h4>
                <ul style={{ margin: 0, padding: 0, listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {brand.lastPosts.slice(0, 3).map((post: any, pIdx: number) => (
                    <li key={pIdx} style={{ fontSize: '0.8rem' }}>
                      <a href={post.url} target="_blank" rel="noreferrer" style={{ 
                        color: 'var(--text-secondary, #444)', 
                        textDecoration: 'none',
                        display: '-webkit-box',
                        WebkitLineClamp: 2,
                        WebkitBoxOrient: 'vertical',
                        overflow: 'hidden'
                      }}>
                        {post.description}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}
      </div>
    </main>
  );
}
